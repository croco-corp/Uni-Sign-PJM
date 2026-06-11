"""Extract poster figures from PJM data.

Picks one signer per (sex, age) bucket from `pjm_data.csv`, takes the middle
frame of the corresponding video, loads the pre-computed RTMPose Wholebody
keypoints, and renders three kinds of figures:

  * spamo_signer.png        — clean middle frame (young M signer)
  * modspamo_signer.png     — clean middle frame (mid-age F signer)
  * modspamo_handcrop.png   — tight crop around the right hand
  * modspamo_landmarks.png  — keypoints rendered on white background
  * unisign_pose_overlay.png — frame with skeleton + hand keypoints overlay

All outputs go to `docs/poster/figures/`. Re-run this script if any pick
needs to change — edit the `PICKS` list at the bottom.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import subprocess
import tarfile
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO         = Path('/home/croco/Uni-Sign')
POSE_DIR     = REPO / 'dataset' / 'PJM' / 'pose_format'   # original training data (RTMPose lightweight, to_openpose=True)
TARS_DIR     = Path('/home/croco/CrocoSign/data/pjm_segments')
INDEX_TSV    = Path('/tmp/pjm_video_index.tsv')
OUT_DIR      = REPO / 'docs' / 'poster' / 'figures'

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- RTMPose Wholebody-134 with to_openpose=True (matches training data) ----
# Body-18 (OpenPose COCO):
#   0=Nose 1=Neck 2=RShoulder 3=RElbow 4=RWrist
#                 5=LShoulder 6=LElbow 7=LWrist
#   8-13 hips/knees/ankles  (skipped — cropped out for seated upper-body video)
#   14=REye 15=LEye 16=REar 17=LEar
# 18-23 feet (skip)
# 24-91 face 68 (skip — too dense)
# 92-112  left hand  (21)
# 113-133 right hand (21)

BODY_LIMBS = [
    (0, 1),               # nose ↔ neck
    (1, 2), (1, 5),       # neck ↔ shoulders
    (2, 3), (3, 4),       # right arm: shoulder → elbow → wrist
    (5, 6), (6, 7),       # left arm:  shoulder → elbow → wrist
    (0, 14), (0, 15),     # nose ↔ eyes
    (14, 16), (15, 17),   # eyes ↔ ears
]
BODY_DOT_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17]   # upper body + face landmarks

HAND_LIMBS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),              # palm
]

LH_RANGE = range(92, 113)
RH_RANGE = range(113, 134)

# colours (BGR for cv2)
COL_BODY = (255, 180, 60)   # cyan-ish
COL_LH   = (90, 70, 230)    # magenta/red
COL_RH   = (60, 200, 240)   # warm yellow
COL_DOT  = (255, 255, 255)
KP_THRESH_BODY = 0.50   # lightweight mode confidence ∈ [0, 1]
KP_THRESH_HAND = 0.40
LINE_PX = 4
DOT_PX  = 5


# -----------------------------------------------------------------------------
def load_index() -> dict[str, str]:
    idx = {}
    with INDEX_TSV.open() as f:
        for line in f:
            v, t = line.rstrip().split('\t')
            idx.setdefault(v, t)
    return idx


def extract_mp4(video_name: str, tar_name: str, dest: Path) -> None:
    with tarfile.open(TARS_DIR / tar_name) as tf:
        member = tf.getmember(video_name)
        with tf.extractfile(member) as src, open(dest, 'wb') as out:
            out.write(src.read())


def get_video_sar(mp4_path: Path) -> float:
    """Return sample-aspect-ratio (pixel-aspect) of the video, 1.0 if square pixels."""
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_streams', '-select_streams', 'v:0', str(mp4_path)],
            capture_output=True, text=True, check=True,
        )
        info = json.loads(out.stdout)['streams'][0]
        sar = info.get('sample_aspect_ratio') or '1:1'
        if sar in ('0:1', '0:0', ''):
            return 1.0
        n, d = (int(x) for x in sar.split(':'))
        return (n / d) if d else 1.0
    except Exception:
        return 1.0


def read_frame(mp4_path: Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'cannot read frame {idx} from {mp4_path}')
    sar = get_video_sar(mp4_path)
    if abs(sar - 1.0) > 0.01:
        h, w = frame.shape[:2]
        new_w = int(round(w * sar))
        frame = cv2.resize(frame, (new_w, h), interpolation=cv2.INTER_LANCZOS4)
    return frame


def pick_best_frame(kp_n: np.ndarray, sc: np.ndarray,
                    min_hand_separation: float | None = None) -> int:
    """Pick the frame with the cleanest upper-body+hands detection.

    Quality criteria:
      1. Minimum confidence over [shoulders, elbows, wrists, hand-bases] >= 0.55
      2. Shoulders at similar Y (|Δy| < 0.05)
      3. Both shoulders in upper 65% of frame
      4. (optional) horizontal distance between hand-bases >= min_hand_separation
         — useful when distinct two-hand visualizations are needed.
    Among passing frames, maximize min confidence + bonus for hand spread.
    """
    essentials = [2, 3, 4, 5, 6, 7, 92, 113]
    T = len(sc)
    best, best_score = -1, -1.0
    for t in range(T):
        sc_min = float(sc[t][essentials].min())
        if sc_min < 0.55:
            continue
        sy_diff = abs(kp_n[t, 2, 1] - kp_n[t, 5, 1])
        if sy_diff > 0.05:
            continue
        if kp_n[t, 2, 1] > 0.65 or kp_n[t, 5, 1] > 0.65:
            continue
        hand_dx = abs(kp_n[t, 92, 0] - kp_n[t, 113, 0])
        if min_hand_separation is not None and hand_dx < min_hand_separation:
            continue
        score = sc_min - 5 * sy_diff + 0.5 * hand_dx
        if score > best_score:
            best_score = score
            best = t
    if best < 0:
        print('  (no clean frame found, falling back to middle)')
        return T // 2
    return best


def load_pose(base: str) -> tuple[np.ndarray, np.ndarray]:
    with (POSE_DIR / f'{base}.pkl').open('rb') as f:
        d = pickle.load(f)
    kp = np.array(d['keypoints'])     # (T, 1, 134, 2) normalized
    sc = np.array(d['scores'])        # (T, 1, 134)
    return kp[:, 0], sc[:, 0]


def to_pixels(kp_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    p = kp_norm.copy()
    p[:, 0] *= w
    p[:, 1] *= h
    return p.astype(np.int32)


def draw_skeleton(canvas: np.ndarray, kp: np.ndarray, sc: np.ndarray,
                  *, draw_body: bool = True, draw_hands: bool = True) -> None:
    if draw_body:
        for a, b in BODY_LIMBS:
            if sc[a] > KP_THRESH_BODY and sc[b] > KP_THRESH_BODY:
                cv2.line(canvas, tuple(kp[a]), tuple(kp[b]), COL_BODY, LINE_PX)
        for i in BODY_DOT_IDX:
            if sc[i] > KP_THRESH_BODY:
                cv2.circle(canvas, tuple(kp[i]), DOT_PX, COL_DOT, -1)
                cv2.circle(canvas, tuple(kp[i]), DOT_PX, COL_BODY, 1)

    if draw_hands:
        for hand_range, col in [(LH_RANGE, COL_LH), (RH_RANGE, COL_RH)]:
            base = hand_range.start
            for la, lb in HAND_LIMBS:
                a, b = base + la, base + lb
                if sc[a] > KP_THRESH_HAND and sc[b] > KP_THRESH_HAND:
                    cv2.line(canvas, tuple(kp[a]), tuple(kp[b]), col, LINE_PX - 1)
            for j in hand_range:
                if sc[j] > KP_THRESH_HAND:
                    cv2.circle(canvas, tuple(kp[j]), DOT_PX - 1, COL_DOT, -1)
                    cv2.circle(canvas, tuple(kp[j]), DOT_PX - 1, col, 1)


def hand_crop(frame: np.ndarray, kp: np.ndarray, sc: np.ndarray,
              hand: str = 'right', pad: float = 0.25,
              output_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Crop a hand region matching production SpaMo hand-ViT extraction:
       bbox of confident hand keypoints, 25% pad on each side, resize+letterbox to 224×224."""
    rng = RH_RANGE if hand == 'right' else LH_RANGE
    pts = [kp[i] for i in rng if sc[i] > KP_THRESH_HAND]
    if not pts:
        raise RuntimeError(f'no confident {hand} hand keypoints')
    pts = np.array(pts)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w, h   = x1 - x0, y1 - y0
    pw = w * pad
    ph = h * pad
    H, W = frame.shape[:2]
    L = int(max(0, x0 - pw))
    R = int(min(W, x1 + pw))
    T = int(max(0, y0 - ph))
    B = int(min(H, y1 + ph))
    cropped = frame[T:B, L:R]

    # letterbox to output_size (preserve aspect, pad with black)
    if cropped.size == 0:
        return cropped
    ch, cw = cropped.shape[:2]
    scale = min(output_size[0] / cw, output_size[1] / ch)
    new_w, new_h = int(cw * scale), int(ch * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((output_size[1], output_size[0], 3), dtype=resized.dtype)
    ox = (output_size[0] - new_w) // 2
    oy = (output_size[1] - new_h) // 2
    canvas[oy:oy+new_h, ox:ox+new_w] = resized
    return canvas


def render_landmarks_white(frame_shape: tuple[int, int],
                           kp: np.ndarray, sc: np.ndarray) -> np.ndarray:
    h, w = frame_shape
    canvas = np.full((h, w, 3), 255, np.uint8)
    draw_skeleton(canvas, kp, sc, draw_body=True, draw_hands=True)
    return canvas


# -----------------------------------------------------------------------------
def write(name: str, img: np.ndarray) -> None:
    out = OUT_DIR / name
    cv2.imwrite(str(out), img, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    print(f'wrote {out}  ({img.shape[1]}×{img.shape[0]})')


def process(label: str, base: str, tar_name: str,
            outputs: list[str], video_idx: dict[str, str],
            min_hand_separation: float | None = None,
            preferred_frame: int | None = None) -> None:
    print(f'\n--- {label} :: {base}  (in {tar_name}) ---')
    with tempfile.TemporaryDirectory() as tmp:
        mp4 = Path(tmp) / f'{base}.mp4'
        extract_mp4(f'{base}.mp4', tar_name, mp4)
        kp_n, sc = load_pose(base)
        if preferred_frame is not None:
            idx = preferred_frame
        else:
            idx = pick_best_frame(kp_n, sc, min_hand_separation=min_hand_separation)
        frame = read_frame(mp4, idx)
        h, w = frame.shape[:2]
        kp = to_pixels(kp_n[idx], w, h)
        s  = sc[idx]
        print(f'  frame {idx}/{len(kp_n)}  shape={w}×{h}')

        if 'clean' in outputs:
            write(f'{label}_signer.png', frame)

        if 'overlay' in outputs:
            ov = frame.copy()
            draw_skeleton(ov, kp, s, draw_body=True, draw_hands=True)
            # tight crop around all visible keypoints + 15% padding
            visible = []
            for i in BODY_DOT_IDX:
                if s[i] > KP_THRESH_BODY:
                    visible.append(kp[i])
            for r in [LH_RANGE, RH_RANGE]:
                for i in r:
                    if s[i] > KP_THRESH_HAND:
                        visible.append(kp[i])
            if visible:
                pts = np.array(visible)
                x0, y0 = pts.min(axis=0)
                x1, y1 = pts.max(axis=0)
                w, h = x1 - x0, y1 - y0
                pad_x = w * 0.15
                pad_y = h * 0.15
                H, W = ov.shape[:2]
                L = int(max(0, x0 - pad_x))
                R = int(min(W, x1 + pad_x))
                T = int(max(0, y0 - pad_y))
                B = int(min(H, y1 + pad_y))
                ov = ov[T:B, L:R]
            write(f'{label}_pose_overlay.png', ov)

        if 'handcrop' in outputs:
            for which in ('left', 'right'):
                try:
                    hc = hand_crop(frame, kp, s, hand=which)
                    if hc.size == 0 or min(hc.shape[:2]) < 30:
                        print(f'  {which} hand crop too small, skipping')
                        continue
                    write(f'{label}_handcrop_{which}.png', hc)
                except RuntimeError as e:
                    print(f'  {which} hand crop failed: {e}')

        if 'landmarks' in outputs:
            lm = render_landmarks_white((h, w), kp, s)
            write(f'{label}_landmarks.png', lm)


PICKS = [
    # label,  base,         tar,                 outputs,        min_hand_sep,  preferred_frame
    ('spamo',    '047_000_M', 'train-000002.tar', ['clean'],                            None, 46),
    ('modspamo', '001_000_M', 'train-000000.tar', ['clean', 'handcrop', 'landmarks'],   0.20, None),
    ('unisign',  '007_000_M', 'train-000000.tar', ['overlay'],                          None, None),
]


def main() -> None:
    idx = load_index()
    for label, base, tar_name, outputs, min_sep, pref in PICKS:
        process(label, base, tar_name, outputs, idx,
                min_hand_separation=min_sep, preferred_frame=pref)
    print(f'\nAll figures in: {OUT_DIR}')


if __name__ == '__main__':
    main()
