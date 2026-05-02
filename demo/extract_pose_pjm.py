import argparse
import glob
import os

os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib/ollama/cuda_v12:' + os.environ.get('LD_LIBRARY_PATH', '')
import tarfile
import tempfile
import pickle
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from rtmlib import Wholebody


def process_frame(frame, wholebody):
    frame = np.uint8(frame)
    keypoints, scores = wholebody(frame)
    H, W, C = frame.shape
    return keypoints, scores, [W, H]


def process_video(video_path, output_path, wholebody, max_workers=16, overwrite=False):
    if os.path.exists(output_path) and not overwrite:
        return

    import cv2
    data = {'keypoints': [], 'scores': []}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"fail to open: {video_path}")
        return

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_frame, f, wholebody) for f in frames]
        results = [f.result() for f in futures]

    for keypoints, scores, w_h in results:
        data['keypoints'].append(keypoints / np.array(w_h)[None, None])
        data['scores'].append(scores)

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tars_dir', required=True)
    parser.add_argument('--tgt_dir', required=True)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--backend', default='onnxruntime', choices=['opencv', 'onnxruntime', 'openvino'])
    parser.add_argument('--mode', default='lightweight', choices=['performance', 'lightweight', 'balanced'])
    parser.add_argument('--max_workers', type=int, default=16)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.tgt_dir, exist_ok=True)

    wholebody = Wholebody(
        to_openpose=True,
        mode=args.mode,
        backend=args.backend,
        device=args.device
    )

    tar_paths = sorted(glob.glob(os.path.join(args.tars_dir, '*.tar')))
    print(f"found {len(tar_paths)} tar files")

    for tar_path in tqdm(tar_paths, desc='tars'):
        with tarfile.open(tar_path) as tar:
            members = [m for m in tar.getmembers() if m.name.endswith('.mp4')]
            for member in tqdm(members, desc=os.path.basename(tar_path), leave=False):
                key = member.name.replace('.mp4', '')
                output_path = os.path.join(args.tgt_dir, key + '.pkl')
                if os.path.exists(output_path) and not args.overwrite:
                    continue

                mp4_bytes = tar.extractfile(member).read()
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                    tmp.write(mp4_bytes)
                    tmp_path = tmp.name

                try:
                    process_video(tmp_path, output_path, wholebody, args.max_workers, overwrite=True)
                finally:
                    os.unlink(tmp_path)


if __name__ == '__main__':
    main()
