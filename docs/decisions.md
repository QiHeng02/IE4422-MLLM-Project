# Decision log

One line per decision that changes the data, the metrics, training or the scope.
The viva will ask "why did you do X?"; the answer should be here.

**Status:** `agreed` = team agreed · `proposed` = suggested, confirm at the next team meeting · `changed` = replaced by a later entry.

| # | Date | Decision | Why | Alternatives considered | Status |
|---|---|---|---|---|---|
| 1 | 15 Oct | Topic: context-aware classroom behaviour analysis (replaces PPE topic) | Another group had the PPE problem statement | Crowd occupancy, workplace safety, misinformation | agreed |
| 2 | 15 Oct | Main user: lecturers and TAs; exam is one lesson type | Fits SCB (lesson photos); rubric marks broad audiences down | Invigilation-first | proposed |
| 3 | 15 Oct | Model: Qwen2.5-VL-3B-Instruct + LoRA | Runs on one A40; pipeline already verified in the PPE work | 7B (if time allows) | agreed |
| 4 | 15 Oct | Scope cut: no CLIP, no self-consistency, no rank sweep, no own NTU photos; QLoRA and face blur optional | 3-week timeline; one GPU runs training jobs one after another | Full original ladder | agreed |
| 5 | 16 Oct | Computer lab is the held-out lesson type (never in train or val) | Tests if the model follows a rule it never trained on (RQ2) | Hold out exam or discussion | proposed |
| 6 | 16 Oct | Split by source video, stratified by SCB subset | Near-identical frames across splits inflate scores; each subset labels different behaviours, so a plain split can drop a behaviour from test | Random image split | proposed |
| 7 | 16 Oct | A behaviour not labelled in an image's SCB subset is ignored when scoring that image | Its absence is unknown, not "no one"; scoring it would punish correct flags | Treat as absent | proposed |
| 8 | 16 Oct | No answer balancing | Balancing breaks the same-image-every-rule pairing that contrastive training and the flip test need | Downsample majority answers | proposed |
| 9 | 16 Oct | Answer boxes in Qwen-resized pixel space; `max_pixels` identical in training and inference | Qwen2.5-VL grounds boxes on the resized image | Normalised 0–1000 boxes | proposed |
| 10 | 16 Oct | LoRA run A uses 2× the epochs of run B | Run B has 2× the examples; matched steps and image exposure leave only the rule pairing different | Same epochs (confounded) | proposed |
| 11 | 16 Oct | Main metric is per-question behaviour F1; box recall is secondary | A 3B model is weak at finding small students in crowded rooms; box-only F1 could be ~0 for every method | Box F1 as main metric | proposed |
| 12 | | Behaviour set after SCB class check (see PROJECT_PLAN day-1 blocker) | | A / B / C | **open** |
| 13 | 16 Oct | Option A ruled out: public SCB has no phone / computer / talk labels | Checked HF subsets and YOLO.zip (training code + teacher-behaviour runs only) | — | agreed |
| 14 | 16 Oct | Available student classes: discuss (5,392 boxes), TurnHead (11,156), BowHead (4,962), hand-raising (13,453), read (24,078), write (9,841) | `inspect_scb.py` on 3 downloaded subsets | — | fact |
| 15 | 16 Oct | Split groups by source video across subsets; `group_regex` = digits before "_", else first 4 digits | Same video IDs appear in several subsets; the 4-digit rule is a guess, verified with `--check-leakage` | group per subset | proposed |
| 16 | | YOLO baseline: use the authors' per-subset weights (`exp/weights/best.pt`) or retrain on our split? | Authors' weights may have trained on images in our test set | retrain YOLOv8 on our train split | **open** |
