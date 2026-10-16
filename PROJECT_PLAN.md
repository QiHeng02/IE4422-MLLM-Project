# Project guide

The team's reference for this project. Read sections 1–3 once to understand the project. Come back to sections 4–12 whenever you work on it.

| Section | Read it when you want to know… |
|---|---|
| [0. Where we are](#0-where-we-are-updated-16-oct) | what's done, what's open, what's next |
| [1. The project in one page](#1-the-project-in-one-page) | what we are doing and why |
| [1b. Behaviours, rules and the rule table](#1b-behaviours-rules-and-the-rule-table) | the terms behind open decisions D1–D3 |
| [2. Scope](#2-scope) | whether something is in, optional or out |
| [3. Research questions](#3-research-questions-and-what-counts-as-a-result) | what we are trying to find out |
| [4. Model](#4-model) | how the model works and what we train |
| [5. Datasets](#5-datasets) | which data, for what, and its status |
| [6. Methods and baselines](#6-methods-and-baselines) | what we compare and why each one is there |
| [7. Pipeline](#7-pipeline-step-by-step) | how to do each step, and when a step is "done" |
| [8. Evaluation](#8-evaluation) | how scores are computed and the rules for fair results |
| [9. Lecture topics](#9-lecture-topics-applied) | which course topic each part applies |
| [10. Timeline and to-do](#10-timeline-and-to-do) | what to do this week and who owns it |
| [11. How to…](#11-how-to) | common tasks: run an experiment, change a rule, add a behaviour |
| [12. Rules that keep results valid](#12-rules-that-keep-results-valid) | what NOT to do |
| [13. Risks and fallbacks](#13-risks-and-fallbacks) | what to do when something breaks |

---

## 0. Where we are (updated 16 Oct)

### Done
- [x] Topic confirmed: no overlap with other groups
- [x] Repo, project guide, scripts and tests (5 passing)
- [x] Cluster: `classroom` conda env, repo cloned, SCB data downloaded to `/projects/IE4422/data/scb`
- [x] SCB checked: class counts (`inspect_scb.py`) and label quality by eye (20 images per class)
- [x] Stakeholder email drafted (sends Sun 18 Oct; **update its rule table first**)

### Decided
- **Two problems, one model:** (1) class attention, (2) invigilation. See [§1](#1-the-project-in-one-page).
- **Phone / computer usage data are not in the public SCB release.** Therefore we use other datasets to add them (§5).

### Open decisions (blocking the pipeline)

| # | Decision | What it means | Needs | Owner |
|---|---|---|---|---|
| D1 | **Behaviour list** | Which student actions the model looks for. See [§1b: Behaviours](#behaviours). | Task 2a result (§5) | team |
| D2 | **Rule types**, and which one is held out | The situations the model is told about, e.g. "closed-book exam". See [§1b: Rule types](#rule-types). | team | team |
| D3 | **Rule table** | For each behaviour × rule type: flag, OK or unused. See [§1b: The rule table](#the-rule-table). Stakeholder confirms it. | D1, D2 | team |
| D4 | **YOLO baseline weights** | Use the SCB authors' YOLO models, or retrain on our split. See [§6](#6-methods-and-baselines) and `decisions.md` #16. | before week 2 | [ ] |

### Next tasks
1. **Task 2a:** check the Roboflow exam_cheating dataset (classes, quality, wide shot or close-up)
2. **Team meeting:** D1–D3, then update `rules.csv`, class map and the stakeholder email
3. **Task 3:** filter → split → build questions (first full pipeline run)
4. **Task 4:** zero-shot Qwen baseline on the cluster

---

## 1. The project in one page

### The problem
In large educational settings, teacher or invigilators cannot continuously monitor dozens of students simultaneously

1. **Class attention.** A lecturer teaching a large class cannot see, while teaching, which students have drifted off-task (on a phone, talking, head down). 
2. **Invigilation.** One invigilator watches dozens of candidates and cannot catch every phone, glance at a neighbour, or hidden note. 

**Current tools fall short.** Detectors give each action **the same label everywhere**. But whether an action is a problem **depends on the situation**:
- Talking is fine in a group discussion, off-task in a lecture.
- Reading notes is cheating in a closed-book exam, fine in an open-book one.

A fixed detector cannot tell these apart unless someone rewrites its rules in code for every situation.

> **Problem statement.** Can a vision-language model take the current rule as plain text, for a lecture or an exam, and flag the students whose visible behaviour breaks it? And does fine-tuning keep the model reading that rule?

*"Attention" here means visible on-task behaviour in one photo. A single photo cannot show attention span or intent, and we never claim it does.*

### Our idea
Give a vision-language model (VLM) **two inputs**: the photo, and the rule for the current situation written as text. The model flags students whose visible behaviour breaks **that** rule. The same photo can have different correct answers under different rules:

| Input 1: photo | Input 2: rule (text) | Correct output |
|---|---|---|
| Student A reading notes, student B looking at a neighbour | "This is a **closed-book** exam." | Flag A (reading notes) and B (looking around) |
| *same photo* | "This is an **open-book** exam." | Flag B only |
| *same photo* | "This is a **group discussion**." | Flag no one |

The output is a box around each flagged student, the behaviour, and a one-line reason. A small script turns it into an annotated photo for a person to review.

### Two applications, one model (agreed 16 Oct)

| | **1. Class attention** | **2. Invigilation** |
|---|---|---|
| User | lecturers and TAs | exam invigilators |
| Rule types | lesson types, e.g. lecture vs group discussion | exam types, e.g. closed-book vs open-book |
| Behaviours (data permitting) | talking, turning head, head down, hand-raising | using phone, looking around, reading notes, talking |
| Output wording | "may need a check", never "not paying attention" | "flag for review", never "cheating" |

Both share one model and one pipeline; only the rule text differs. Invigilation gets extra exam data (phone, looking around, reading notes), so its experiments go deeper. Having both lesson and exam rules is also what makes the flip test meaningful.

### The research question
When we fine-tune the VLM with LoRA, does it **keep reading the rule**, or learn to answer from the picture alone? We also test a fix, **rule-contrastive training**: every training photo appears with every rule and its different answer.

### Why it matters
If the model reads the rule, an invigilator or lecturer adds a new rule by writing one sentence: no new code, no retraining. If it ignores the rule, it is just an expensive detector.

**Any result is a finding.** The course marks what we try, observe and explain, not whether the method wins.

---

## 1b. Behaviours, rules and the rule table

These three ideas are what open decisions D1–D3 (§0) are about.

### Behaviours
**What:** the visible student actions the model looks for, e.g. using phone, turning head, talking.
**Why we must choose:** each behaviour needs **human-drawn boxes** in our data; those boxes are the correct answers we train and score on. A behaviour with no labels cannot be used. §5 lists what we have.
**What makes a good one:** good label quality, and an answer that **changes with the rule** (see the rule table below).

### Rule types
**What:** the situations we tell the model about, written as one sentence, e.g. "This is a closed-book exam." We need about 3.
**Held-out rule:** one rule type is **never used in training**. We only test on it, to see whether the model follows a rule it has never seen (the unseen-rule test, RQ2).

### The rule table
**What:** a grid that says, for each behaviour in each rule type, whether it should be flagged:

| Behaviour | Closed-book exam | Open-book exam | Group discussion |
|---|---|---|---|
| Reading notes | flag | OK | OK |
| Looking around | flag | flag | OK |

*(Example only; the team fills in the real table, D3.)* Each cell is `flag`, `OK` or `unused` (unclear, not scored). It is stored in `configs/rules.csv`.

**Why we need it:** the datasets only say *what* a student is doing ("reading notes"), not whether it is wrong. The table adds "wrong in this situation". It is used to:
1. **make the correct answers** for every photo + rule (training and testing),
2. **run the YOLO baseline**, which applies the table as an if-statement,
3. **check requirements with the stakeholder** (round 1).

**The key constraint:** a behaviour only tests rule reading if its row **changes across rule types**. A row that is `flag` everywhere (or `OK` everywhere) gives the same answer whatever the rule, so it cannot show whether the model reads the rule. Such rows can stay, but they don't help the flip test.

---

## 2. Scope

| | Item |
|---|---|
| **In (must do)** | Single photos · ~4–6 behaviours (D1) × 3 rule types (D2), covering invigilation + class attention · Qwen2.5-VL-3B · zero-shot, few-shot, LoRA (normal + rule-contrastive) · YOLO + rule table baseline · flip test, unseen-rule test, reworded rules · SCB + one exam-behaviour dataset (§5) · one second test set · 2 stakeholder rounds · annotated-photo output |
| **Optional (only if week 2 ends early)** | QLoRA · face-blur test · describe-then-reason prompt · second training seed · video-frame demo |

**Rule of thumb.** If a task does not answer RQ1–RQ3 or serve a rubric row, it is out. Ask the team before adding scope.

---

## 3. Research questions and what counts as a result

| RQ | Question | Experiment | Possible findings |
|---|---|---|---|
| **RQ1** | Does LoRA beat prompting? | zero-shot vs few-shot vs LoRA on known rules (`known_f1`) | LoRA higher → training teaches classroom cues. Similar → prompting is enough for this task. |
| **RQ2** | Does LoRA keep the model reading the rule? Does rule-contrastive training help? | flip test, unseen-rule test, reworded rules | Normal LoRA drops on `unseen_f1` or `flip_acc` vs zero-shot → fine-tuning made it ignore text. Contrastive recovers it → our fix works. No drop → fine-tuning is safe here. |
| **RQ3** | When is a VLM worth it over a detector? Does it transfer? | YOLO + rule table vs Qwen; best model on a second test set | YOLO wins known rules → detectors are better at finding objects. VLM wins new rules without code → its value is flexibility. Second-set drop → domain shift. |

**Headline claim, honest form.** "We test whether LoRA fine-tuning keeps a VLM reading exam and lesson rules, and whether rule-contrastive training protects it." Never claim more than the numbers show.

---

## 4. Model

**Qwen2.5-VL-3B-Instruct**, fine-tuned with LoRA.

How one question flows through the model:

| Step | Input | Part | What happens | Trained? |
|---|---|---|---|---|
| 1 | Photo | **Vision encoder** | Turns the image into image tokens | ❄️ frozen |
| 2 | Image tokens | **Projector** | Maps image tokens into the language model's space | ❄️ frozen |
| 3 | Rule + question (text) | **Text tokenizer** | Turns the words into text tokens | — |
| 4 | Image + text tokens | **Qwen2.5 language model** | Image and text tokens attend to each other in every layer (joint fusion) and write the answer | ❄️ frozen, **plus LoRA adapters 🔥 (the only trained part)** |
| 5 | Model text | **JSON answer** | `flagged` list (box + behaviour) and a `reason` | — |
| 6 | JSON | **Display script** | Draws boxes and a one-line summary for the reviewer | — (not part of the model) |

| Part | Detail | Why |
|---|---|---|
| Base model | Qwen2.5-VL-3B-Instruct | Fits one A40 (46 GB); has built-in box grounding; already verified in our earlier pipeline |
| Trained part | LoRA, rank 8, on language-model linear layers | Small and fast; the vision side already sees well |
| Frozen | Vision encoder, projector | Keeps the test about rule reading, not vision |
| Fusion type | Joint: image and text tokens attend to each other in every layer | Contrast with YOLO (late fusion) |
| Input | photo (image) + lesson rule and question (text) | 2 modalities, as the rubric requires |
| Output | JSON: `flagged` list of `{bbox_2d, behaviour}` + `reason` | A script can score it exactly |
| Box coordinates | Pixels of the image **after Qwen's resize** | That is how Qwen2.5-VL grounds boxes. `build.max_pixels` must match in training and inference. |
| Decoding | Greedy (temperature 0) | Repeatable results |
| Licence | Qwen Research License (non-commercial) | Fine for coursework |

---

## 5. Datasets

### Sources

| Dataset | Role | Behaviours it labels | Licence | Status (16 Oct) |
|---|---|---|---|---|
| **SCB, Hugging Face release** | train, val, main test | discuss, TurnHead, BowHead, hand-raising, read, write | academic, non-commercial | ✅ downloaded and checked |
| **Roboflow [exam_cheating](https://universe.roboflow.com/behavior-cheating/exam_cheating-keaor)** | extra data with **real exam scenes** | Use_phone, Look_around, Cheat_paper, Normal (~1,000 images) | CC BY 4.0 | ⏳ not checked (quality, wide shot vs close-up). Task 2a. |
| **DIPSER** | second test only | none of ours: attention/emotion labels | CC BY 4.0 | not started; we hand-label 100–200 frames |
| HCCB / other Roboflow sets | backup | varies | check | only if needed |

SCB classes `talk`, `using the phone` and `using the computer` appear in the SCB paper but are not in the public release. Phone comes from Roboflow; no public source with human labels for computer use was found, so computer use is dropped.

### Behaviours: what we can label

| Behaviour | Source | Label quality (20-image check) | Application |
|---|---|---|---|
| Talking (`discuss`) | SCB | 20/20 | both |
| Turning head (`TurnHead`) | SCB | 16/20 ⚠️ weakest | both |
| Head down (`BowHead`) | SCB | 19/20 | attention |
| Hand-raising | SCB | 19/20 | attention |
| Reading / writing | SCB | not checked | both (reading notes) |
| Using phone | Roboflow | not checked | invigilation |
| Looking around | Roboflow | not checked | invigilation |
| Reading notes (`Cheat_paper`) | Roboflow | not checked | invigilation |

Counts in the HF release: discuss 5,392 boxes · TurnHead 11,156 · BowHead 4,962 · hand-raising 13,453 · read 24,078 · write 9,841.

### Known data problems, and how we handle them

| Problem | Handling |
|---|---|
| Each subset or dataset labels only its own classes | `filter_scb.py` records `labelled` per image; scoring ignores unlabelled behaviours |
| Every Discuss image contains discussion: no "no one talking" examples | open: accept as limitation, or hand-check ~50 other images as negatives |
| TurnHead labels ~80% correct | keep; state as limitation |
| Frames from one video look almost identical; the same video appears in several subsets | split groups by video **across subsets** (`group_regex`); `--check-leakage` finds near-twins |
| SCB shows lessons, not exams | exam rules on lesson photos are "imagined"; Roboflow adds real exam scenes |
| Two photo styles if we add Roboflow (possibly close-ups) | flip test still valid (same photo, different rules); report style as a limitation |
| SCB includes school-age students and Chinese classrooms | state as a limitation; second test set measures transfer |

Download commands: `data/README.md`. Data is never committed.

---

## 6. Methods and baselines

All methods answer the **same test questions** and are scored by the **same script**.

| Method | Reads rule text? | Why it is here | How to run |
|---|---|---|---|
| **YOLO + rule table** | No; the rule table is code | Current approach (late fusion). Answers "why not a detector + if-statement?" Needs a hand-written column for every new lesson type. | `yolo_rules_baseline.py` |
| **Qwen zero-shot** | Yes | Starting point. Also the reference for RQ2: did LoRA make rule-reading worse than this? | `run_vlm.py` |
| **Qwen few-shot** (2 examples) | Yes | Best result without training (RQ1) | `run_vlm.py --few-shot 2` |
| **Qwen + LoRA, normal** (run A) | Yes | Standard fine-tuning: each training image with one random rule | `lora_single.yaml` |
| **Qwen + LoRA, rule-contrastive** (run B, ours) | Yes | Each training image with every training rule → forces rule reading | `lora_contrastive.yaml` |
| *Optional:* QLoRA | Yes | GPU memory saved vs accuracy lost | add `quantization_bit: 4` to a LoRA config |

**Fair comparison of runs A and B.** Run B has 2× the examples, so run A uses 2× the epochs. Both take about the same steps and see each image equally often. Only the rule pairing differs.

---

## 7. Pipeline step by step

Every script reads `configs/default.yaml`. Change settings there, never in code.

### Step 0. Check the data
- **Why:** confirm the classes exist before building on them.
- **Run:** `python scripts/inspect_scb.py --root data/raw/scb`
- **Output:** `results/data_check/scb_inspect.md` (classes, counts, sample file names)
- **Done when:** behaviour set decided (§0 D1); `scb_class_map` and `split.group_regex` set. *(SCB part done 16 Oct; `group_regex` set.)*

### Step 1. Filter
- **Why:** keep only behaviours a lesson rule can allow or forbid.
- **Run:** `python scripts/filter_scb.py`
- **Output:** `data/processed/manifest.jsonl` (one line per image: size, subset, labelled behaviours, boxes)
- **Done when:** every behaviour in `rules.csv` has boxes (no WARNING); ~50 images checked by eye.

### Step 2. Rule table
- **Why:** SCB says *what* a student does, not whether it is wrong. The table adds "wrong in this lesson". It makes the correct answers, drives the YOLO baseline, and is what the stakeholder confirms.
- **Edit:** `configs/rules.csv` (cells: `flag`, `ok`, `unused` = ambiguous, not scored)
- **Done when:** stakeholder round 1 confirms it; change logged in `docs/stakeholder_notes.md`. **Freeze it after this.**

### Step 3–4. Split and make questions
- **Why:** a VLM needs question + answer pairs; splitting by video stops leakage; the held-out rule stays out of training for the unseen-rule test.
- **Run:** `python scripts/split_data.py --check-leakage` then `python scripts/build_questions.py`
- **Output:** `data/processed/questions/`: `train_single`, `train_contrastive`, `val`, `test`, `test_reworded` (.jsonl)
- **Done when:** leakage report reviewed; both subsets appear in test; positive rate per lesson type noted in the experiment log.

### Step 5. Baselines
- **Run:** `run_vlm.py` (zero-shot, few-shot) on **val**; `yolo_rules_baseline.py`
- **First time on the GPU:** add `--limit 20` and check the answers by eye.
- **Done when:** val scores in `results/summary.csv`; few-shot setting chosen.

### Step 6. Train LoRA
- **Run:** `python scripts/export_llamafactory.py`, then `llamafactory-cli train configs/llamafactory/lora_single.yaml` and `.../lora_contrastive.yaml`
- **Pick the checkpoint** by val `flag_f1` (run `run_vlm.py --adapter ...` on val), not by training loss.
- **Done when:** both adapters saved; peak GPU memory and training time logged.

### Step 7. Evaluate
- **Run:** each final method once on `test.jsonl` and `test_reworded.jsonl`, then `evaluate.py` for each.
- **Done when:** results table filled; ~50 mistakes sorted by type (missed flag, wrong behaviour, wrong rule, bad box).

### Step 8. Display
- **Run:** `python scripts/visualize.py --questions ... --predictions ... --n 10`
- **Output:** annotated photos with a one-line summary ("may need a check", never "cheating")
- **Done when:** shown in stakeholder round 2; feedback recorded and applied.

---

## 8. Evaluation

### Metrics (all from `scripts/evaluate.py`)

| Metric | What it measures | Used for |
|---|---|---|
| `flag_f1` (main) | Per question: did it name the right behaviours to flag? F1 over (question, behaviour) pairs | all RQs |
| `known_f1` / `unseen_f1` | `flag_f1` on trained rules / on the held-out rule only | RQ1 / RQ2 |
| `flip_acc` | Same image, all 3 rules: share of images where every answer is right | RQ2 (does it read the rule?) |
| `f1_wording_para1/2` | `flag_f1` on reworded rules | RQ2 (meaning, not exact words) |
| `box_recall` (secondary) | Flagged students found at IoU ≥ 0.5 | localisation, reported but not headline |
| `valid_json` | Share of answers that parse | invalid answers count as "no one flagged" |

**Why F1, not accuracy.** Most behaviour checks are "not flagged". A model that never flags anything would get high accuracy and zero F1.

### Protocol

1. Choose prompts, checkpoints and settings on **val** only.
2. Run the **test** set once per final method. Do not tune after looking at test scores.
3. One scorer (`evaluate.py`) for every method.
4. Every reported number comes from `results/summary.csv`, with `--note` saying the checkpoint or seed.

### Results table (fill from `results/summary.csv`)

| Method | flag_f1 (SCB) | flip_acc | unseen_f1 | reworded F1 | flag_f1 (2nd test set) |
|---|---|---|---|---|---|
| YOLO + rule table | | | | | |
| Qwen zero-shot | | | | | |
| Qwen few-shot | | | | | |
| Qwen + LoRA, normal | | | | | |
| Qwen + LoRA, rule-contrastive | | | | | |

---

## 9. Lecture topics applied

| Topic | Week | Where in the pipeline | What it tests |
|---|---|---|---|
| Instruction tuning | 4 | step 3: image + rule + question → JSON | teaches the task and an exact output format |
| LoRA | 4 | step 6 | does cheap fine-tuning help, and does it keep rule reading? |
| QLoRA (optional) | 4 | step 6 | memory saved vs accuracy lost |
| Prompting: few-shot | 6 | step 5 | how far prompts go without training |
| Fusion: late vs joint | 5, 7 | step 5 vs 6 | detector + rules vs VLM reading rule text |
| Trust and evaluation | 8 | step 7: reworded rules, val/test discipline, F1 | does it follow meaning? are scores fair? |

---

## 10. Timeline and to-do

Put owner initials in the brackets. Tick boxes in a pull request. **Submission deadline: [fill in]** · **Presentation: [fill in]**

### Week 1 (16–22 Oct): data and baselines

**Milestone:** frozen question files and zero-shot val scores.

- [x] Confirm no other group has the topic
- [x] Download SCB (HF), run `inspect_scb.py`, check labels by eye
- [x] Set `split.group_regex` from SCB file names
- [ ] Confirm topic and two-application framing with instructor [ ]
- [ ] Assign an owner to every line [ ]
- [ ] **Task 2a:** check Roboflow exam_cheating [ ]
- [ ] Team meeting: decisions D1–D3 (§0); log in `decisions.md` [ ]
- [ ] Update rule table in the stakeholder email; **send by Sun 18 Oct** [ ]
- [ ] Stakeholder round 1 reply; freeze `rules.csv`; notes [ ]
- [ ] **Task 3:** `filter_scb.py` → `split_data.py --check-leakage` → `build_questions.py` [ ]
- [ ] **Task 4:** GPU env (`requirements-gpu.txt`); `run_vlm.py --limit 20`; zero-shot on val [ ]
- [ ] YOLO baseline: decide D4; prepare weights [ ]
- [ ] Second test set: DIPSER access, or pick a backup [ ]
- [ ] Cited figures for proposal Section 1 (class sizes; invigilator-to-student ratios) [ ]

### Week 2 (23–29 Oct): training and tests

**Milestone:** every method scored on test; stakeholder round 2 done.

- [ ] Few-shot on val [ ]
- [ ] YOLO + rule table on test [ ]
- [ ] Export data; LoRA run A (normal) [ ]
- [ ] LoRA run B (rule-contrastive) [ ]
- [ ] Pick checkpoints on val; run all methods on `test` and `test_reworded` [ ]
- [ ] Hand-label 100–200 frames of the second test set; write its manifest [ ]
- [ ] `visualize.py` examples; stakeholder round 2; notes [ ]

### Week 3 (30 Oct – 5 Nov): results and write-up

**Milestone:** proposal submitted; viva rehearsed.

- [ ] Best model on the second test set; final results table [ ]
- [ ] Error analysis (~50 mistakes by type) [ ]
- [ ] Apply round 2 feedback; proposal Section 8 [ ]
- [ ] Optional extras only if week 2 ended early (§2) [ ]
- [ ] Final proposal (≤ 2,000 words); verify every reference [ ]
- [ ] Slides; viva rehearsal: every member answers questions [ ]
- [ ] Peer evaluation on NTULearn (compulsory, 5% penalty) [everyone]

---

## 11. How to…

**Run and record an experiment**
1. `git pull`; note the commit hash.
2. Run the method (`run_vlm.py` / `yolo_rules_baseline.py`) → `results/predictions/<method>_<split>.jsonl`.
3. `python scripts/evaluate.py --method <method> --questions <file> --predictions <file> --note "<checkpoint, seed>"`
4. Add an entry to `docs/experiment_log.md`: command, commit, result, **what you observed**.
5. Commit `results/summary.csv`, `results/metrics/`, and the log.

**Change the rule table** (only before it is frozen, or after a stakeholder request)
1. Edit `configs/rules.csv`.
2. Re-run `build_questions.py`. Earlier results are no longer comparable: re-score them.
3. Log it in `docs/decisions.md` with the reason.

**Add or change a behaviour**
1. Add a row to `rules.csv`; map the SCB class in `scb_class_map`.
2. Re-run steps 1, 3–4. Log the decision.

**Add a new lesson type** (e.g. for the stakeholder)
1. Add a column to `rules.csv` and an entry under `contexts` in `default.yaml` (canonical + 2 paraphrases).
2. To test it as unseen, add it to `heldout_contexts`, not `train_contexts`.

**Add another dataset (e.g. Roboflow, or a second test set)**
1. If it is YOLO format with a `.yaml` listing `names` and `images/` + `labels/` folders: put it under `data/raw/scb/<name>/`, map its class names in `scb_class_map`, and re-run steps 1, 3–4. Nothing else changes.
2. Otherwise, write a small script that outputs the **same manifest format** as `filter_scb.py` (image, width, height, subset, labelled, boxes).
3. For a test-only set: put all its images in test in a splits file.

---

## 12. Rules that keep results valid

| Don't | Because |
|---|---|
| Tune anything on test scores | the score is then optimistic and indefensible at the viva |
| Train on held-out-rule questions or paraphrased wordings | they are the unseen-rule and reworded tests |
| Change `max_pixels` between training and inference | boxes will be in the wrong pixel space |
| Compare runs with different `rules.csv` or questions | differences come from data, not the method |
| Report numbers from a terminal or notebook | only `results/summary.csv` is traceable |
| Commit data, images, weights or tokens | licences and repo size |
| Write "cheating" or "not paying attention" in outputs | we see behaviour, not intent; the wording was agreed with stakeholders |
| Claim "first to…" without checking | examiners probe absolute claims |

---

## 13. Risks and fallbacks

| Risk | Sign | Fallback |
|---|---|---|
| No usable phone data (Roboflow quality poor) | Task 2a fails | invigilation uses talking, turning head, reading notes; phone moves to future work |
| Too few boxes for a behaviour | filter WARNING, or < ~100 boxes | drop or merge it; log the decision |
| `group_regex` cannot find source videos | split prints "almost one group per image" | rely on `--check-leakage`; group near-twins by hand |
| VLM answers are not valid JSON | `valid_json` < 0.9 | add one example to the prompt; check `max_new_tokens` |
| Box recall near 0 for every method | crowded rooms, small students | keep `flag_f1` as the headline; report boxes as a limitation |
| GPU queue or memory problems | OOM, long waits | batch size 1 + more gradient accumulation; QLoRA; Colab as backup |
| LoRA run late | week 2 slipping | drop optional items; train run B first (it is our method) |
| DIPSER unavailable or too slow to label | no access by mid week 2 | backup dataset (§5), or ~100 labelled frames |
| SCB authors' YOLO weights trained on our test images | leakage check (D4) | retrain YOLOv8 on our train split (a few GPU hours, before LoRA) |

---

## Key files

| File | What it is |
|---|---|
| `configs/default.yaml` | every setting: paths, class map, lesson wordings, split, model |
| `configs/rules.csv` | the rule table |
| `configs/llamafactory/*.yaml` | LoRA runs A and B |
| `docs/decisions.md` | why we did each thing (for the viva) |
| `docs/stakeholder_notes.md` | meeting notes and changes made |
| `docs/experiment_log.md` | what was run and what we observed |
| `results/summary.csv` | the only source of reported numbers |
