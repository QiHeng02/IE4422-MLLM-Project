# Audio branch: build the clean transcript table

Follows Hao Yu's collaboration design (step B5). Written for WTD; if the team picks another dataset, only the
input step changes. Laptop only: no cluster, no audio model (the dataset already has transcripts).

**Task.** Turn the dataset's transcripts into ONE clean table that `scripts/fuse_wtd.py` reads.
The talk statistics and the talk network are computed later by `fuse_wtd.py`, not here.

## Input (WTD)
- Human transcripts: `Group_XX_Oracle.csv` with `Utterance, Start, End, Group, Participant, Transcript`
- Automatic transcripts: `Group_XX_Google.csv` (check its columns first: does it have a speaker ID?)
- Seat map from Qi Heng: `data/processed/wtd/seats.csv` with `group, participant, seat` (seat = P1 left, P2 middle, P3 right in the video)
Ask Qi Heng for the CSV files.

## Output: `data/processed/wtd/transcripts.csv`
Exactly these columns: `group, speaker, start, end, text, source`
- `group`: integer (1–10)
- `speaker`: **P1 / P2 / P3 from the seat map** (not the raw participant number: participant 1 is not always on the left)
- `start`, `end`: seconds from the video start
- `source`: `human` (Oracle) or `asr` (Google)

## Rules
- **Drop participant 4** (the experimenter reading the instructions).
- Keep one row per utterance; do not merge.
- If the Google file has no speaker IDs, output `human` rows only and tell the group.

## Done when
- `transcripts.csv` exists with the columns above; no participant 4; speakers only P1/P2/P3.
- Row count per group = Oracle rows minus participant-4 rows (WTD: e.g. Group_02 = 181 − 20 = 161).
- 5 utterances checked against the video: do the times line up?
- Script committed as `scripts/build_transcripts.py`.
