# Faza 3 – raport implementacji (PJM full fine-tune)

**Data:** 2026-04-19
**Branch:** `PJM-adaptation`
**Cel:** odpalić pełny fine-tune (visual + mT5) na PJM MS split,
 żeby wyjść z mode-collapse obserwowanego w fazie 2.

---

## TL;DR – co zrobiłem

1. **Dodałem guard w `fine_tuning.py`** – test eval leci tylko w ostatniej
   epoce (dev eval nadal co epokę → wandb zbiera krzywą uczenia).
2. **Wpiąłem `--quick_break N` w `train_one_epoch`** – wcześniej flaga
   była parsowana ale nigdzie nieużywana. Teraz faktycznie przerywa epokę
   po N krokach. Użyte w preflight dry-run.
3. **Final test eval teraz jest logowany** – wynik `evaluate(phase='test')`
   w ostatniej epoce leci do wandb jako `final_test_*` i do `log.txt`.
   Wcześniej return value był odrzucany.
4. **Napisałem `script/train_pjm_phase3_ms.sh`** – full fine-tune,
   start z `csl_stage2_weight.pth`, `--lr 3e-4`, `--weight-decay 0.01`,
   `bs=8 × grad_accum=2`, 15 epok, MS split.
5. **Napisałem `script/queue_phase3.sh`** z preflight dry-run:
   grzecznie ubija fazę 2 (SIGTERM → 60 s grace → SIGKILL),
   odpala 5-krokowy test (OOM check), dopiero potem właściwy trening.
6. **NIE dodawałem** `--lr_visual` / differential LR – precedens z repo
   (`train_stage2.sh`, `train_stage3.sh`) to jednolite `lr 3e-4`
   dla całego modelu naraz. Prostota > hipotetyczne zyski.
7. **NIE ładuję phase-2 checkpointu** jako init – phase 2 nauczyła się
   wypluwać "I, I, I, I..." (mode collapse), ten bias zatrułby fazę 3.
   Start z CSL-News base jest czystszy.

Kod jest w branchu, nic jeszcze nie odpaliłem – faza 2 dalej chodzi
(PID 773061, epoka ~3 z 10). Czekam na Twój sygnał, żeby ją zabić
i wystartować fazę 3.

---

## Co zmieniłem w kodzie

### 1. `fine_tuning.py` – pomiń test eval poza ostatnią epoką

**Przed:**
```python
if utils.is_main_process():
    test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
    evaluate(args, test_dataloader, model, model_without_ddp, phase='test')
```

**Po:**
```python
if utils.is_main_process():
    test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
    if epoch == args.epochs - 1:
        final_test_stats = evaluate(args, test_dataloader, model, model_without_ddp, phase='test')
        if args.wandb and WANDB_AVAILABLE:
            wandb.log({f'final_test_{k}': v for k, v in final_test_stats.items()})
        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps({f'final_test_{k}': v for k, v in final_test_stats.items()}) + "\n")
```

**Dlaczego:**
- Faza 2 liczyła pełen test eval co epokę (3–4 min × 10 epok = ~35 min
  samego testu). W fazie 3 (15 epok) to by się skumulowało.
- Dev eval nadal wykonuje się co epokę – *on* jest potrzebny do krzywej
  walidacyjnej w wandb i do wyboru `best_checkpoint.pth` przez
  porównanie `test_stats["bleu4"]`.
  (Tak, ta zmienna nazywa się `test_stats` ale zawiera wartości z dev –
  to dziedzictwo z oryginalnego kodu. Nie dotykałem.)
- Test eval tylko na koniec daje finalny raport liczb, które trafiają
  na papier. Capture-and-log zapewnia, że nie zostaną utracone
  w ostatniej chwili 22-godzinnego runu (pierwotna wersja odrzucała
  return value – silent data loss).

**Uwaga techniczna:** edycja `fine_tuning.py` jest bezpieczna dla
aktualnie biegnącej fazy 2. Python importuje moduł raz – uruchomiony
proces ma skompilowany bytecode w pamięci i nie odczyta zmian z dysku.
Dopiero kolejne `python fine_tuning.py` załaduje nową wersję.

---

### 2. `script/train_pjm_phase3_ms.sh` – faza 3 training

```bash
deepspeed --include localhost:0 --master_port 29514 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --epochs 15 \
  --warmup-epochs 1 \
  --opt AdamW \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --finetune out/csl_news_stage2/csl_stage2_weight.pth \
  --dataset PJM \
  --task SLT \
  --pjm_split ms \
  --bertscore --num_examples 5 \
  --output_dir out/pjm_phase3_full_ms \
  --wandb --wandb_project uni-sign-pjm --wandb_dir ./wandb_logs
```

#### Argumenty – dlaczego takie wartości

**`--batch-size 8`**
- Faza 2 miała `bs=16` i mieściła się w VRAM, ale trenowała tylko LoRA
  (≈1M parametrów mT5 + zero gradów w visual).
- Faza 3 rozmraża WSZYSTKO: STGCN, `proj_linear`, `pose_proj`, pełen
  mT5 (≈580M params). Dla każdego parametru Adam trzyma:
  `param + grad + m + v` × fp32 = 16 bajtów/param ≈ **9.3 GB** samych
  stanów optymalizatora (ZeRO-2 sharduje je, ale mamy 1 GPU →
  shardowanie między 1 GPU = brak oszczędności).
- Aktywacje też rosną, bo teraz backward przepuszcza gradienty przez
  cały graf.
- `bs=8` to bezpieczny punkt startowy, zmierzony na oko z analogii do
  `train_stage3.sh` (oryginalny repo, który też używa `bs=8`).

**`--gradient-accumulation-steps 2`**
- Efektywny batch = 8 × 2 = **16** (matching fazy 2).
- Dzięki temu LR dynamics są porównywalne – gdyby efektywny batch się
  zmienił, `lr 3e-4` mogłoby się zachowywać inaczej.
- `grad_accum` jest "darmowy" pamięciowo: liczysz dwa forward/backward
  po `bs=8`, akumulujesz gradienty, jeden step optymalizatora.

**`--epochs 15`**
- Faza 2 miała 10 epok ale tam trenujesz 1M parametrów (szybko się
  zbiegnie jeśli da się). Tu trenujesz 580M, więc potrzeba więcej
  kroków na stabilizację.
- Oryginalny `train_stage3.sh` używa 20 epok na CSL_Daily.
- 15 to kompromis: dajemy modelowi szansę nauczyć się PJM, ale nie
  tracimy 18h okna GPU na dogorywanie po stabilizacji.

**`--warmup-epochs 1`**
- Przez pierwszą epokę LR rośnie liniowo od 0 do 3e-4, potem kosinusowy
  decay do `min-lr = 1e-8` w epoce 15.
- Warmup jest *szczególnie ważny* gdy startujesz z pretrained ckpt –
  duży LR od razu na już wytrenowane wagi może wywalić model w stronę
  losowego szumu.
- 1 epoka warmupu = ~3000 kroków → wystarczająco długi rozruch.

**`--opt AdamW`**
- Jawnie ustawione (default też jest `adamw`, ale wolę być
  eksplicytny).
- `AdamW` różni się od `Adam` tym, że weight decay jest dodawany do
  parametrów, nie do gradientów – lepsze przy dużym WD.

**`--lr 3e-4`**
- **Uniform LR, bez `--lr_visual`.** Precedens: `train_stage2.sh`
  i `train_stage3.sh` w repo Uni-Sign oba używają `3e-4` dla całego
  modelu naraz. Skoro CSL-News stage 2 został wytrenowany przy `3e-4`
  na wszystkim, kontynuacja przy tym samym LR to najbezpieczniejszy
  wybór.
- Rozważałem differential LR (niższy dla mT5 bo bardziej "delikatny"),
  ale:
  - Repo Uni-Sign tego nie robi.
  - Dodaje kod do utrzymania (param groups, osobne LR w optimizer).
  - Phase 2 collapsed NIE z powodu LR – z powodu zamrożonego visual.
    Odmrożenie visual samo w sobie rozwiązuje problem.

**`--weight-decay 0.01`**
- Oryginalny repo ma `weight_decay=0.0001` jako default.
- Podniosłem do `0.01`, bo:
  - Full fine-tune z 580M parametrów ma duży potencjał do
    overfitting na 24k samples PJM.
  - `0.01` to standard dla Adam/AdamW w fine-tune transformerów
    (HuggingFace defaults, BERT, T5).
  - Kara za duże wagi pomaga utrzymać visual blisko pretrainedu
    CSL-News (regularization do startowego punktu).

**`--finetune out/csl_news_stage2/csl_stage2_weight.pth`**
- **Start z CSL-News stage 2, NIE z phase 2 checkpointu.**
- Phase 2 nauczyła dekoder wypluwać "I, I, I, I..." – gdyby to był init,
  faza 3 musiałaby najpierw odkleić mode collapse *a potem* uczyć się
  PJM. Zbędna komplikacja.
- Phase 2 LoRA-wagi są zapisane (`out/pjm_phase2_lora_ms/*.pth`) ale
  nie chcę ich używać. Gdybyś chciał kiedyś spróbować:
  `peft`'s `merge_and_unload()` wkleiłby je z powrotem do `mt5_model`,
  potem osobny zapis. Nie warto tracić czasu teraz.

**`--pjm_split ms`**
- Multi-speaker split (random 24k/3k/3k). Zgodnie z Twoją decyzją:
  `si` pomijamy, dopóki MS się nie zacznie uczyć.
- `dataset_path` jest zbudowany w `fine_tuning.py` jako
  `split_train_ms.csv` / `split_val_ms.csv` / `split_test_ms.csv`
  (suffix `_ms`).

**Brak `--lora`, brak `--freeze_visual`**
- Brak LoRA: trenujemy wszystkie wagi mT5 bezpośrednio.
- Brak freeze: STGCN + fuser + `pose_proj` uczą się też. To jest cała
  pointa fazy 3.

**`--bertscore --num_examples 5`**
- BERTScore: semantic similarity (a nie tylko n-gram BLEU).
  Pomocne, bo PJM to inny język niż CSL → BLEU może być niski nawet
  przy dobrej treści.
- `num_examples=5`: co ewaluacja drukuje 5 par (ref, hyp) – widać
  czy model w ogóle produkuje sensowne tokeny, czy dalej "I I I".

**`--wandb --wandb_project uni-sign-pjm`**
- Loguje metryki live. Ten sam projekt co faza 2, osobny run
  (`name=os.path.basename(args.output_dir) = pjm_phase3_full_ms`).
- Możesz porównać krzywe faza 2 vs faza 3 w wandb.

---

### 2b. `train_one_epoch` – honor `--quick_break`

```python
if args.quick_break > 0 and (step + 1) >= args.quick_break:
    print(f"[quick_break] stopping epoch after {step+1} steps (quick_break={args.quick_break})")
    break
```

Flaga `--quick_break` była zdefiniowana w `utils.py` ale nigdy nie
konsumowana w `fine_tuning.py`. Dodałem faktyczny break po N krokach.
Używam tego w preflight dry-run (5 kroków) żeby odpowiedzieć na
pytanie "czy mieści się w VRAM" bez marnowania całego GPU okna.

---

### 3. `script/queue_phase3.sh` – orchestracja startu

Kluczowe fragmenty:

```bash
# SIGTERM najpierw (grace), dopiero potem SIGKILL
pkill -TERM -P "$p2_pid"
kill -TERM "$p2_pid"
pkill -TERM -f "fine_tuning.py"

for i in $(seq 1 60); do
  if ! pgrep -f "fine_tuning.py" > /dev/null; then break; fi
  sleep 1
done

if pgrep -f "fine_tuning.py" > /dev/null; then
  pkill -KILL -f "fine_tuning.py"
fi
```

**Dlaczego SIGTERM przed SIGKILL:**
- SIGKILL (`kill -9`) ubija natychmiast – jeśli DeepSpeed był w trakcie
  zapisu checkpointu, plik zostanie uszkodzony.
- SIGTERM pozwala procesowi złapać sygnał i wyjść po zakończeniu
  aktualnego I/O. DeepSpeed nie ma customowego handlera ale ZeRO
  zapisuje ckpt atomicznie w kawałkach – SIGTERM między blokami = czyste
  wyjście. SIGTERM w środku `torch.save()` = niedokończony plik, ale
  nie uszkadza poprzednich.
- 60 s grace to prawie na pewno za dużo (`torch.save` dla ~1.2GB ckpt
  trwa ~5s), ale lepiej dmuchnąć na zimne.

**`pkill -TERM -P "$p2_pid"`**: ubija dzieci procesu, nie tylko sam
proces. `queue_phase2.sh` ma trzy poziomy: `queue → train_ms.sh →
deepspeed → python fine_tuning.py`. `-P` sięga tylko po dzieci, więc
dodatkowo `pkill -TERM -f "fine_tuning.py"` dla pewności dosięga wnuka.

**`nvidia-smi --query-gpu=memory.used,memory.free`** przed startem fazy
3: sanity check że VRAM jest faktycznie zwolniony.

---

## Co się NIE zmienia (a mogło się wydawać, że tak)

**`get_requires_grad_dict` bug** (models.py) – ten bug jest znany: funkcja
zapisuje wszystkie parametry zamiast tylko `requires_grad=True`. W
fazie 2 robi to z checkpointów 1.2 GB (zamiast ~5 MB jak powinno być
dla LoRA). W fazie 3 `requires_grad=True` dla prawie wszystkich
parametrów, więc checkpoint będzie ~1.2 GB tak czy siak – bug jest
efektywnie no-op i nie warto go teraz naprawiać.

**`label_smoothing 0.2`** – oryginalny default. Redukuje overconfidence
dekodera. Zostawiłem.

**`--max_length 256`** – liczba klatek pose. Default, zostawiony.

---

## Ryzyka, które pozostają

### 1. OOM przy starcie (najwyższe ryzyko, MITIGATED)
Phase 3 zje dużo więcej VRAM niż phase 2 (wszystkie grady + Adam states).

**Kolejka to łapie automatycznie:** `queue_phase3.sh` przed właściwym
15-epoka runem odpala 5-krokowy dry-run w `/tmp/phase3_dryrun_$$`.
Jeśli on OOMnie, kolejka wypisuje ostatnie 80 linii logu i kończy
z `exit 2` bez startu głównego treningu. Preflight zajmuje ~6 min
(5 kroków ~1 min + dev + final-test eval które wciąż się odpalą bo
`epochs=1` → `epoch==args.epochs-1` → test też). Tanim kosztem
unikamy scenariusza "OOM po godzinie i 22h okna zmarnowane".

**Plan B jeśli preflight zafail:**
1. Obniż `--batch-size 4 --gradient-accumulation-steps 4` (eff. 16)
   w `script/train_pjm_phase3_ms.sh`.
2. Jeśli dalej OOM: dodaj `--offload` (ZeRO offload do CPU) – wolniej,
   ale się mieści.

Preflight używa tej samej konfiguracji co główny run (`bs=8`, `ga=2`,
`lr 3e-4`) więc jeśli przejdzie, główny run też się zmieści.

### 2. LR 3e-4 może być zbyt agresywne na visual
Visual był trenowany na CSL-News. PJM to inne pozy (różne proporcje
dłoni, inne konwencje). Przy `lr 3e-4` pierwsze kilka kroków może
"popsuć" visual zanim warmup i WD zdążą stabilizować.

**Gdybym miał 2 próby, zacząłbym od `lr 1e-4`.** Ale skoro mamy 1
próbę i precedens repo to 3e-4 – idziemy z 3e-4.

**Sygnał w logach w ciągu pierwszej godziny:**
- Loss < 10 po 500 krokach = model się uczy, normalnie.
- Loss oscyluje 12–20 po 1000 krokach = problem, `lr` zbyt duży lub
  visual rozjechany.
- Loss konstant ~12 = dekoder ignoruje encoder (znowu mode collapse) →
  trzeba sprawdzić czy gradienty w visual faktycznie są niezerowe.

### 3. Mode collapse może wrócić
Jeśli faza 3 też wypluje "I, I, I...", winowajcą nie jest freeze tylko
coś głębszego (label smoothing 0.2, beam search, krzywa zbieżności
dekodera vs enkodera). Wtedy trzeba by:
- Sprawdzić `cross_attention` wagi – czy dekoder w ogóle "patrzy" na
  enkoder.
- Obniżyć `label_smoothing` do 0.1 lub 0.
- Zweryfikować, że `pose_proj` gradienty nie są zerowe.

---

## Jak to odpalić

Kiedy dasz zielone światło:

```bash
# 1. Sanity check: jeszcze raz że wiesz, co się kryje
cat script/train_pjm_phase3_ms.sh
cat script/queue_phase3.sh

# 2. Odpal kolejkę w tle:
nohup bash script/queue_phase3.sh > phase3_queue.log 2>&1 &
echo $! > phase3_queue.pid
disown

# 3. Monitoruj:
tail -f phase3_queue.log
# albo
tail -f out/pjm_phase3_full_ms/log.txt
# albo w wandb: run `pjm_phase3_full_ms` w projekcie `uni-sign-pjm`
```

Kolejka:
1. Ubije fazę 2 grzecznie (SIGTERM, 60 s grace, ewentualnie SIGKILL).
2. Wyświetli `nvidia-smi` żebyś widział że GPU wolny.
3. Odpali fazę 3 MS.
4. Po zakończeniu: checkpointy w `out/pjm_phase3_full_ms/`, final
   test eval w log.txt jako ostatni JSON.

---

## Szacowany czas

- Faza 2 (LoRA, bs=16, 10 epok): ~45 min/epokę × 10 = ~7.5h.
- Faza 3 (full, eff. bs=16, 15 epok):
  - Krok forward+backward ~2× wolniejszy od fazy 2 (bo gradienty przez
    visual).
  - Eval dev co epokę (~3 min), test tylko ostatnia (~3 min).
  - **Szacunek: 90 min × 15 epok ≈ 22h**.

Jeśli masz 18h – możesz skrócić na `--epochs 12` (szacunek ~18h) albo
odpalić 15 epok i obciąć gdzieś w połowie jeśli widać konwergencję.
Best_checkpoint zapisywany on-the-fly po każdej poprawie BLEU-4 na dev,
więc nawet przerwanie nie traci najlepszego modelu.

---

## Files touched

```
modified:   fine_tuning.py          # guard test eval + capture final test + honor quick_break
new file:   script/train_pjm_phase3_ms.sh
new file:   script/queue_phase3.sh
new file:   docs/PHASE3_IMPLEMENTATION_REPORT.md   # ten plik
```

Faza 2 jeszcze żyje – czekam na Ciebie z killem.
