# Audio branch: Whisper + one clean transcript table

**Audio owner.** Same as HY's Remaining Work Guide V2, **Step 3** (Wed 21 – Thu 22 Oct). Laptop only, no cluster.
If this file and the V2 guide disagree, `docs/decisions.md` decides; ask in the group chat.

**Task.** Run our own Whisper on the group videos, then build ONE clean table (`transcripts.csv`) that
`fuse_wtd.py` reads. Talk statistics and the talk network are computed later by `fuse_wtd.py`, not here.

## Before you start
1. Laptop setup from the V2 guide (B3): repo cloned, `.venv` active, `pip install -e .`, `ffmpeg -version` works.
2. **Scripts:** HY's 7 WTD scripts are shared in the group chat, **not on GitHub**. Copy all 7 into `scripts/`
   (`get_wtd.py, check_wtd.py, wtd_frames.py, run_whisper.py, build_transcripts.py, fuse_wtd.py, plot_wtd.py`).
   Use the latest version posted; if HY posts a new one, replace yours. **Never `git add .`**: it would commit them.
3. From the **Data owner** (Step 1): the CSVs → `data/raw/wtd/csv/`, the 3 videos → `data/raw/wtd/video/g01.mp4 … g03.mp4`.
4. From `WTD_handoff`: `data/processed/wtd/seats.csv` → same path in your repo copy.

## Run
```bash
git pull
pip install openai-whisper
python scripts/check_wtd.py seats          # must end with ALL OK

# Whisper, group 1 first (first run downloads the Whisper model; time it)
python scripts/run_whisper.py --video data/raw/wtd/video/g01.mp4 --group 1 --oracle data/raw/wtd/csv/Group_01_Oracle.csv
# then groups 2 and 3
python scripts/run_whisper.py --video data/raw/wtd/video/g02.mp4 --group 2 --oracle data/raw/wtd/csv/Group_02_Oracle.csv
python scripts/run_whisper.py --video data/raw/wtd/video/g03.mp4 --group 3 --oracle data/raw/wtd/csv/Group_03_Oracle.csv

# one table: Oracle = human, our Whisper = asr
python scripts/build_transcripts.py --input data/raw/wtd --seats data/processed/wtd/seats.csv
```
If a video has no sound track, ffmpeg stops with an error: post it, then build with WTD's own Whisper file
instead (`--asr whisper` on `build_transcripts.py`). Google transcripts are **not** used.

## Output: `data/processed/wtd/transcripts.csv`
Columns: `group, speaker, start, end, text, source`
- `speaker`: P1 / P2 / P3 from the seat map (P1 = left, P2 = middle, P3 = right in the video)
- `start`, `end`: seconds from the start of `gNN.mp4` (the one clock everyone uses)
- `source`: `human` (Oracle) or `asr` (our Whisper)
- Participant 4 (the experimenter) is dropped. Whisper can't tell voices apart, so each Whisper segment takes the
  speaker of the Oracle line it overlaps most.

## Check
```bash
python scripts/check_wtd.py transcripts               # must end with ALL OK
python scripts/check_wtd.py spot --group 1 --source asr --n 5
```
- Group 2 should have **161 human rows** (181 Oracle lines − 20 from the experimenter).
- `spot` prints 5 Whisper lines: find each in the video and check the words, the time and the seat.

## Hand off
Put `data/processed/wtd/transcripts.csv` and the folder `data/raw/wtd/whisper_ours/` in `WTD_handoff` (same paths).
Post: **"DONE audio: transcripts.csv ALL OK, human rows g1/g2/g3 = …, asr rows = …, 5/5 spot checks match"**.

Nothing from this step goes in git (data). A wrong format goes back to its owner: don't patch other people's files.
