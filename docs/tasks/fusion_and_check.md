# Fusion and check: vision + transcript → per-student report

**Scripts and fusion owner.** Same as HY's Remaining Work Guide V2, **Step 4** (Thu 22 – Fri 23 Oct). Laptop only.
If this file and the V2 guide disagree, `docs/decisions.md` decides; ask in the group chat.

**Task.** Run `fuse_wtd.py` 4 times (LoRA / zero-shot × human / asr transcript), compare with WTD's collaboration
codes, make one figure per group, and log it.

## Before you start
1. Laptop setup from the V2 guide (B3): repo cloned, `.venv` active, `pip install -e .`.
2. **Scripts:** HY's 7 WTD scripts are shared in the group chat, **not on GitHub**. Copy all 7 into `scripts/`
   (latest version posted). **Never `git add .`**: it would commit them.
3. Copy these from `WTD_handoff` to the same paths in your repo copy:
   - `data/processed/wtd/questions.jsonl` (Model owner, Step 2)
   - `results/predictions/wtd_lora.jsonl` and `results/predictions/wtd_zeroshot.jsonl` (Model owner, Step 2)
   - `data/processed/wtd/transcripts.csv` (Audio owner, Step 3)
   - and `All_Groups_CPS.csv` from the Data owner → `data/raw/wtd/csv/`

## Run
```bash
git pull

# 1. check what you received (a PROBLEM goes back to its owner)
python scripts/check_wtd.py predictions --file results/predictions/wtd_lora.jsonl
python scripts/check_wtd.py predictions --file results/predictions/wtd_zeroshot.jsonl
python scripts/check_wtd.py transcripts

# 2. the four fusion runs (seconds each)
python scripts/fuse_wtd.py --predictions results/predictions/wtd_lora.jsonl     --source human --name wtd_lora_human
python scripts/fuse_wtd.py --predictions results/predictions/wtd_lora.jsonl     --source asr   --name wtd_lora_asr
python scripts/fuse_wtd.py --predictions results/predictions/wtd_zeroshot.jsonl --source human --name wtd_zeroshot_human
python scripts/fuse_wtd.py --predictions results/predictions/wtd_zeroshot.jsonl --source asr   --name wtd_zeroshot_asr

# 3. one results table (paste into docs/experiment_log.md)
python scripts/check_wtd.py results

# 4. compare with WTD's collaboration codes
python scripts/check_wtd.py cps --name wtd_lora_human

# 5. report figures (talk network + per-student evidence under the AAC&U Teamwork rows)
python scripts/plot_wtd.py --name wtd_lora_human --group 1
python scripts/plot_wtd.py --name wtd_lora_human --group 2
python scripts/plot_wtd.py --name wtd_lora_human --group 3
```

## Outputs
- `results/metrics/wtd_*.json`: vision "talking" vs who really spoke (precision, recall, F1)
- `results/wtd/wtd_*_students.csv`: per-student report; `results/wtd/wtd_*_edges.csv`: talk network
- `results/wtd/<run>_gNN.png`: one figure per group

## How to read it
- **Expected pattern (from the test set):** LoRA finds talking but rarely follows a new sentence (participation
  F1 0.12); zero-shot follows new sentences but rarely finds talking. Human vs asr shows how much Whisper's
  mistakes change the numbers.
- **CPS comparison:** speaking share and turns next to each student's share of CPS-coded utterances, plus one rank
  correlation. Talking is not collaborating, so a weak match is a finding, not a bug: write it down either way.
- **Box check (Step 2's count out of 20):** if many boxes were on the wrong seat, say the per-student vision numbers
  (attending, hand-raising, phone) are unreliable and report only frame-level talking.

## Hand off
1. Write the experiment log entry (template at the top of `docs/experiment_log.md`): the commands, the results
   table, the CPS comparison, Step 2's box-check count, what you noticed.
2. Commit and push **only** `results/metrics/wtd_*.json` and `docs/experiment_log.md` (`git status` first).
3. Put `results/wtd/` (CSVs and figures) in `WTD_handoff`.
4. Post: **"DONE fusion: 4 runs, table in experiment_log, 3 figures in hand-off"**.

Wording in everything you write: "evidence for review", "may need a check"; never "attention", "cheating" or a grade.
