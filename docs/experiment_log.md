# Experiment log

One entry per run. Numbers live in `results/summary.csv`; this file records what you ran and what you noticed.
Observations matter as much as scores: the course marks what we learn from each attempt.

## Template

```
### YYYY-MM-DD · <method> · <who>
Command:      python scripts/... (full command)
Config:       configs/default.yaml @ commit <hash>; seed
Questions:    val / test / test_reworded
GPU / time:   A40, peak memory, wall time
Result:       flag_f1 = , flip_acc = , unseen_f1 = , valid_json =
Observed:     what went right / wrong; 2–3 example answers worth noting
Next:         what this changes
```

## Runs
### 2026-10-21 · WTD collaboration extension: late fusion · <your name>
Command:      scripts/fuse_wtd.py x4 (lora/zeroshot x human/asr); check_wtd.py results; check_wtd.py cps --name wtd_lora_human; plot_wtd.py --group 1/2/3
Config:       configs/default.yaml @ commit 7f98027; held-out group_work sentence; one frame every 2 s; decisions 42–46
Questions:    WTD groups 1–3, 1,534 frames (4,602 student-frames); trained LoRA = outputs/lora_single_v3
GPU / time:   model runs by <Model owner> on A40; fusion on laptop, seconds per run
Result:       vision "talking" vs transcript (precision / recall / F1):
              LoRA + human 0.276 / 0.425 / 0.334 · LoRA + Whisper 0.240 / 0.397 / 0.299
              zero-shot + human 0.304 / 0.513 / 0.382 · zero-shot + Whisper 0.270 / 0.489 / 0.348
              valid JSON 1.0 for all; box check 55/60 boxes on the right student (4 hidden behind objects, 1 missed)
              CPS check (wtd_lora_human): Spearman 0.95 between speaking share and share of CPS-coded utterances, 9 students;
              68–87% of each student's utterances CPS-coded; 0 off-topic codes in groups 1–3
Observed:     - Students speak in ~33% of student-frames, so random guessing gives precision ~0.33. Both models are below
                that: single frames cannot show who is talking. Audio is needed for speaking; vision only adds gaze/phone/hands.
              - Without the transcript, ~1,700 student-frames would count as participation while the student was silent.
              - LoRA does not beat zero-shot on this held-out setting (F1 0.33 vs 0.38), consistent with participation F1 0.12 on test.
              - Whisper instead of human transcripts lowers F1 by only 0.03–0.04.
              - Talk volume tracks expert CPS coding closely in these on-task lab groups; the off-topic case cannot be tested here.
              - Unbalanced groups visible: g1 P1 57% vs P3 17% speaking; g2 P3 14% speaking but 86% of own talk CPS-coded.
Next:         Step 5 review; figures g01–g03 to stakeholder (Part C); limitations: single-frame vision, lab task, 9 students.