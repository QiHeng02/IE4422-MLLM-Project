# Audio branch: transcripts → per-student participation stats

**Task.** A Python script that turns a group's transcript into per-student participation stats and a
"who replies to whom" network, saved as CSV files (+ one picture per group).
Laptop only: no cluster, no audio model (the transcripts already say who spoke and when).

**Tools.** `pip install pandas networkx matplotlib`

## Input
One transcript CSV per group with: speaker ID, start (s), end (s), text.
WTD example: `Utterance, Start, End, Group, Participant, Transcript`. Drop non-students (WTD: participant 4).

## Definitions
- **utterance** = one transcript row.
- **turn** = consecutive utterances by the same student, no other student in between.
- **speaking_time** = sum of (end − start) of the student's utterances (s).
- **talk_share** = student's speaking_time / all students' speaking_time in the group.
- **reply** = student B's turn starts within **5 s** after student A's turn ends (B ≠ A) → edge A → B.
- **window** = 30-s slices from the start (0–30, 30–60, …); an utterance belongs to the window it starts in.

## Outputs (`results/audio/<dataset>/`)
1. `audio_students.csv`: `dataset, group, student, speaking_time_s, talk_share, n_utterances, n_turns, mean_turn_s, replies_given, replies_received, partners_replied_to`
2. `audio_windows.csv`: `dataset, group, student, window_start_s, speaking_time_s, n_utterances, spoke`
3. `audio_edges.csv`: `dataset, group, from_student, to_student, n_replies`
4. `figures/<group>_network.png`: arrows A → B, thickness = n_replies

## Done when
- One command runs all groups (e.g. `python scripts/audio_branch.py --input <folder>`).
- Each group's talk_share sums to 1.0; utterance counts match the transcript (students only).
- One group spot-checked by hand.
- Script committed as `scripts/audio_branch.py`.

**Optional:** run it on the automatic transcript too (WTD: `Group_XX_Google.csv`, if it has speaker IDs) and compare.
