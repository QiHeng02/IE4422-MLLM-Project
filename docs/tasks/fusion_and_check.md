# Fusion and check: vision + transcript → per-student report

Follows Hao Yu's collaboration design (steps B6–B7). Written for WTD. Laptop only.

**Task.** Run `scripts/fuse_wtd.py` (Hao Yu) on the vision answers and the clean transcript, then check the results.

## Inputs
- `data/processed/wtd/questions.jsonl` (from `scripts/wtd_frames.py`, Qi Heng)
- Vision answers: `results/predictions/wtd_lora.jsonl` and `wtd_zeroshot.jsonl` (Qi Heng, cluster)
  The model's own output: per box one of talking / turning head / hand raising / using phone.
- `data/processed/wtd/transcripts.csv` (audio branch)

## Run (4 times)
```
python scripts/fuse_wtd.py --predictions results/predictions/wtd_lora.jsonl     --source human --name wtd_lora_human
python scripts/fuse_wtd.py --predictions results/predictions/wtd_zeroshot.jsonl --source human --name wtd_zeroshot_human
python scripts/fuse_wtd.py --predictions results/predictions/wtd_lora.jsonl     --source asr   --name wtd_lora_asr
python scripts/fuse_wtd.py --predictions results/predictions/wtd_zeroshot.jsonl --source asr   --name wtd_zeroshot_asr
```
(Change the `--source` choices in the script from `whisper` to `asr`: WTD's automatic transcript is Google.)

## Outputs
- `results/metrics/<name>.json`: vision "talking" vs transcript (precision, recall, F1), false "talking" caught by audio
- `results/wtd/<name>_students.csv`: per-student report
- `results/wtd/<name>_edges.csv`: talk network (who replies to whom)

## Checks
1. **Before trusting anything:** look at 20 frames with their boxes. Does the box-to-person mapping (left / middle / right) hold?
   If boxes miss people, report talking per frame ("someone is talking") instead of per person.
2. **Against WTD's collaboration codes** (`All_Groups_CPS.csv`, same rows as the Oracle transcript): per student, compare
   speaking share / turns / replies with the share of the student's utterances that carry collaboration codes.
   **Use proportions, not counts** (students who talk more always have more coded utterances).
3. Expect from the test results: LoRA finds talking but never hand-raising and may ignore the new sentence;
   zero-shot finds hand-raising but not talking. Report both.

## Done when
- 4 runs done; one table: vision talking F1 (LoRA vs zero-shot, human vs ASR transcript).
- One per-student report + talk network figure for one group, for the proposal.
- Each run written in `docs/experiment_log.md` with its command.
