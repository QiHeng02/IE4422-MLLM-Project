# Fusion and check: combine talk + vision, map to AAC&U, test against labels

**Task.** Combine the audio branch output with the vision output per student, map it to two AAC&U Teamwork
rubric rows, and check it against the dataset's collaboration labels. Laptop only.

## Inputs
- `audio_windows.csv`, `audio_students.csv`, `audio_edges.csv` (audio branch)
- `vision_windows.csv` (Mambo, from the cluster): `dataset, group, student, window_start_s, n_frames, n_frames_participating, visible_share`
- The dataset's collaboration labels (WTD: `All_Groups_CPS.csv`, same rows as the transcript)

## 1. Agree the mapping rules first (with the team, before looking at any result)
| AAC&U row | Evidence | Rule |
|---|---|---|
| Contributes to team meetings | talk_share, n_turns, visible_share | (team decides) |
| Facilitates the contributions of team members | replies_given, partners_replied_to, replies_received | (team decides) |

Also decide: what counts as "participating" in a window when audio and vision disagree.

## 2. Fusion
Join audio and vision on `dataset, group, student, window_start_s` →
`results/fusion/<dataset>/students_summary.csv` (per student: audio features, vision features, two rubric scores),
plus one example "report card" for the proposal.

## 3. "Truth" from the labels: use proportions, not counts
(Students who talk more always have more coded utterances.)
- share of the student's utterances with any collaboration code;
- share per facet (WTD: CONST, NEG, MAINTAIN);
- for "Facilitates": share coded as responding to / questioning others (WTD: `CPS_NEG_Responds_*`).

## 4. Compare
For each rubric row: **audio only, vision only, audio + vision** → Spearman rank correlation with the truth
across all students (state how many). Also: is the top student per group the same by score and by labels?

## Done when
- Rules agreed before step 4.
- `students_summary.csv`, one results table (3 versions × 2 rubric rows), one report card.
- Caveats noted: few students; correlation is not the rubric itself (AAC&U is judged by people).
- Scripts committed (`scripts/fusion.py`, `scripts/check_labels.py`).
