# PJM Transfer Learning Plan

Punkt startowy: checkpoint pretrenowany na CSL-News (1985h). Niskopoziomowe reprezentacje pose/ręka/twarz są transferowalne między językami migowymi. Adnotacje PJM są po angielsku — brak potrzeby zmian języka w modelu (identyczny schemat jak How2Sign).

## Faza 1: Zero-shot / minimal fine-tune (1-2 dni)

- Załaduj CSL-News pretrained checkpoint
- Odpal inference na PJM dev set bez żadnego treningu (zmień tylko target language prompt na English — już obsługiwany)
- Zmierz BLEU → absolutny floor, diagnostyka ile model rozumie visual signal bez uczenia PJM

## Faza 2: LoRA fine-tune na mT5 decoder (~tydzień, 1 GPU)

- Zamroź visual encoder (pose fuser + RGB encoder)
- Dodaj LoRA do mT5 decoder: rank 16-32, target_modules = q_proj, v_proj, (opcjonalnie o_proj)
- Trenuj ~10-30 epok na PJM train set
- Najpierw multi-speaker split (łatwiejsze), potem signer-independent
- Porównaj z SpaMo baselinem — spodziewana zauważalna poprawa signer-independent

## Faza 3: Full fine-tune visual + decoder (~2-3 tygodnie)

- Odblokuj visual encoder z bardzo małym LR (10× mniejszym niż dla dekodera)
- Ostrożna regularyzacja — ryzyko overfittingu na 150 signerach
- Największy spodziewany zysk wydajnościowy

## Faza 4 (opcjonalna, najbardziej ambitna): Continue pretraining na PJM

- Przed fine-tuningiem SLT: dodatkowy stage pretrainingu metodą pose-text alignment na PJM
- Pipeline: CSL-News → PJM-pretraining → PJM-SLT
- Wymaga par sign-text bez potrzeby konkretnych splitów
- Teoretycznie najlepsze wyniki
