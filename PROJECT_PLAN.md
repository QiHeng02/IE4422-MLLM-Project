# Project guide

The team's reference for this project. Read Section 0–3 once to understand the project. Use Section 7 and Section 10 when you work on a task.

**If documents disagree:** the code and `configs/` show what is implemented. `docs/decisions.md` records what we decided and why. This guide must follow both; fix it when it does not.

| Section | Read it when you want to know… |
|---|---|
| [0. Status](#0-status-16-oct) | what is done, what is decided, what is still open |
| [1. The project in one page](#1-the-project-in-one-page) | what we are doing and why |
| [1b. Key ideas](#1b-key-ideas) | question files, settings, behaviours, the flag table |
| [2. Scope](#2-scope) | whether something is in or optional |
| [3. Research questions](#3-research-questions) | what we are trying to find out, and what each result would mean |
| [4. Model](#4-model) | how the model works and what we train |
| [5. Datasets](#5-datasets) | which data, for what, its status and its problems |
| [6. Methods and baselines](#6-methods-and-baselines) | what we compare and why |
| [7. Pipeline](#7-pipeline-step-by-step) | what each step does, what it makes, and when it is done |
| [8. Evaluation](#8-evaluation) | how we score, and the rules for fair results |
| [9. Lecture topics](#9-lecture-topics-applied) | which course topic each part applies |
| [10. Timeline and dependencies](#10-timeline-and-dependencies) | what to do this week, in what order, and who owns it |
| [11. How to…](#11-how-to) | common tasks |
| [12. Rules that keep results valid](#12-rules-that-keep-results-valid) | what NOT to do |
| [13. Risks and fallbacks](#13-risks-and-fallbacks) | what to do when something breaks |

---

## 0. Status (16 Oct)

### Done
- [x] Topic confirmed: no overlap with other groups.
- [x] Repo with project guide, scripts and tests. `pytest -q`: 5 passed (16 Oct).
- [x] Cluster: `classroom` conda env, repo cloned, SCB data in `/projects/IE4422/data/scb`.
- [x] SCB checked: box counts per class (`inspect_scb.py`), and 20 photos each for 4 classes by eye (discuss, TurnHead, BowHead, hand-raising). `read` and `write` are not checked.
- [x] Stakeholder email drafted (not in the repo). It still shows the old rule table, so update it before it goes out on Sun 18 Oct.

### Decided
- **Two problems, one model:** class attention and invigilation (`decisions.md` #17).
- **Approach A** (`decisions.md` #22): one model for both settings; the text tells it which setting it is in. Research questions in Section3. agreed on 16 Oct. **The team confirms on Sat 17 Oct.**
- **Phone / computer usage data are not in the public SCB release. Therefore we use other datasets to add them.** Phone comes from Roboflow exam_cheating. We found no labelled data for computer use, so we drop it (#21).

### Open decisions

Each decision blocks a pipeline step (Section7). **D1–D3 decided 17 Oct (#12, #27–#29); D6 proposed (#24).** The team confirms them.

| # | Decision | Why it matters | Blocks | Needs first |
|---|---|---|---|---|
| D1 | **Behaviour list** (see [Section1b](#behaviours)). ✅ **17 Oct:** talking, turning head (SCB TurnHead + Roboflow Look_around), hand-raising, using phone, reading notes. Head down dropped (#12) | Only behaviours with boxes drawn by people can be scored | Step 1 (filter), D3 | done |
| D2 | **Settings** (see [Section1b](#settings)). ✅ **17 Oct:** lecture and closed-book exam (trained); participation (held out, test only) (#27, #29) | RQ2 only works if some answers change between settings | Step 2 (flag table), step 4 (questions) | done |
| D3 | **Flag table** (see [Section1b](#the-flag-table)). ✅ **17 Oct:** `configs/rules.csv`; `scripts/check_rules.py` passes (#28) | It makes every answer key | Step 4 | done. The team freezes it; Dr Lim reviews example outputs in round 2 |
| D4 | **YOLO baseline**: train one YOLO on our training photos, or use the SCB authors' models | The authors' models are 3 separate models (one per SCB subset, older YOLO format with `exp/opt.yaml`). None covers the Roboflow classes, and our script loads one model. Their training photos may also be in our test set. **Recommendation: train our own.** | Step 5 (YOLO) | before 19 Oct |
| D5 | **Third test dataset** (DIPSER): keep or drop? | SCB + Roboflow already give two datasets. DIPSER needs 100–200 photos labelled by hand. **Recommendation: optional.** | Week 2 labelling work | team |
| D6 | **How to split Roboflow photos.** **Proposal (#24), code ready:** our own split by scene (file-name rules + picture fingerprint); the 2 huge scenes go to train only, every 5th photo kept | Task 2a found ~136 real scenes in 772 photos, and Roboflow's own split leaks | Step 3 (split) | team confirms |

---

## 1. The project in one page

### The problem
In large classes and exam halls, a teacher or an invigilator cannot watch dozens of students at the same time.

1. **Class attention.** A lecturer teaching a large class cannot see, while teaching, which students have drifted off-task (on a phone, talking, head down). *[add a cited figure: typical NTU lecture size]*
2. **Invigilation.** One invigilator watches dozens of candidates and cannot catch every phone, glance at a neighbour, or hidden note. *[add a cited figure: invigilator-to-candidate ratio]*

**Current tools fall short.** Detectors give each action **the same label everywhere**. But whether an action is a problem **depends on the situation**:
- Talking is fine in a group discussion, off-task in a lecture.
- Reading notes is cheating in a closed-book exam, fine in an open-book one.

A fixed detector cannot tell these apart unless someone rewrites its rules in code for every situation.

> **Problem statement.** Can one vision-language model serve both lecturers and invigilators, switched only by a text instruction that says the setting? Does LoRA fine-tuning improve it without making it ignore that instruction?


### Our idea
The model gets **two inputs**: a photo, and one sentence that says the setting. It flags the students whose visible behaviour does not fit **that** setting. The same photo can have different correct answers:

| Input 1: photo | Input 2: setting (text) | Correct output (from our flag table, D3) |
|---|---|---|
| Student A talking, student B turning head | "This is a **lecture**." | Point out A only (turning to the board or a screen is normal) |
| *same photo* | "This is a **closed-book exam**." | Point out A and B |
| Student C on a phone, student D reading notes | "This is a **closed-book exam**." | Flag C and D |
| *same photo* | "This is a **lecture**." | Flag C only (reading is fine in a lecture) |

The output is a box around each flagged student, the behaviour, and a one-line reason. A small script draws this on the photo for a person to review.

### Two applications, one model

| | **1. Class attention** | **2. Invigilation** |
|---|---|---|
| User | lecturers and TAs | exam invigilators |
| Settings | lesson types, e.g. lecture, group discussion | exam types, e.g. closed-book, open-book |
| Behaviours (D1) | talking, using phone | talking, turning head (looking around), using phone, reading notes |
| Output wording | "may need a check", never "not paying attention" | "flag for review", never "cheating" |

Both use one model and one pipeline; only the setting text differs. The research questions are in [Section3](#3-research-questions).

**Held-out test (#29):** a third setting, *participation* ("the teacher is checking which students are taking part"), is **never trained**. The answer is hand-raising and talking. It tests whether the text alone can switch the model to an application it never saw.

### Why it matters
- **One tool for two jobs.** The text decides the task, so the text input does real work (our main multimodal point).
- **New situations need one sentence, not new code**, but only if the model really reads the text. RQ2 checks this.
- **A wrong flag in an exam can harm a student.** RQ3 measures how often that happens.

**Any result is a finding.** The course marks what we try, observe and explain, not whether the method wins.

---

## 1b. Key ideas

### Question files: how the model learns and is tested
The model learns and is tested like a student: with **questions and answer keys**.
- **One question** = one photo + one setting sentence + "Which students should be flagged?"
- **Its answer key** = the boxes that people drew on that photo, kept only for behaviours the flag table marks `flag` in that setting.
- A script (step 4) writes these into files. The **training** file teaches the model. The **validation** file helps us choose settings during the work. The **test** file gives the final scores, once.

### Settings
**What:** the sentence that tells the model where it is, e.g. "This is a lecture." Each setting is a column in the flag table.
**Decision D2 (17 Oct):** lecture and closed-book exam are trained. Participation is held out.
**Held-out setting:** a setting that we never use in training. Testing on it shows whether the model follows a setting it never saw. Expect LoRA to do worse than prompting here: in training, hand-raising is never pointed out.
**Rule for the sentences:** name the situation **and its purpose** (who is checking for what, #37), never the behaviours. E.g. "This is a closed-book exam. The invigilator is checking for possible cheating." The question is neutral ("Consider only these behaviours: … Whether a behaviour should be pointed out depends on the situation. Which students should be pointed out in this situation, if any?", #36) because participation points out good behaviour, and so that the list is a scope, not a target.

### Behaviours
**What:** the visible student actions the model looks for, e.g. using phone, turning head, talking.
**Why we must choose (D1):** each behaviour needs **boxes drawn by people** in our data. Those boxes are the answer keys. A behaviour without boxes cannot be scored. Section5 lists what we have.
**A good behaviour:** labels that are mostly correct, and a flag that **changes between settings** (next section).

### The flag table
**What:** a grid that says, for each behaviour in each setting, `flag`, `ok` or `unused` (unclear, not scored). It is stored in `configs/rules.csv`.

| Behaviour | Lecture | Closed-book exam | Participation (held out) | Why |
|---|---|---|---|---|
| Talking | flag | flag | flag | Off-task in a lecture, possible cheating in an exam; taking part in a discussion |
| Turning head | ok | flag | ok | Following the teacher or a screen; looking around in an exam |
| Hand-raising | ok | ok | flag | Never wrong; the sign of taking part |
| Using phone | flag | flag | ok | Off-task or cheating; not a sign of taking part |
| Reading notes | ok | flag | ok | Normal in a lecture; not allowed in a closed-book exam |

*(D3, decided 17 Oct. Head down was dropped: often note-taking.)*

**Why we need it:** the datasets only say *what* a student does ("head down"), not whether it is wrong. The table adds "wrong in this setting". It is used to:
1. **make the answer keys** for every photo + setting (step 4),
2. **run the YOLO baseline**, which applies the table as an if-statement (step 5),
3. **check with the stakeholder** what should be flagged (round 1).

**Key constraint for RQ2:** RQ2 asks whether the model uses the text or guesses the setting from the photo. To test this, **some rows must change between the lesson and exam columns, for photos from both datasets.** In our table, turning head (SCB TurnHead photos) and reading notes (Roboflow photos) do this. Talking, hand-raising and using phone do not, so the Discuss and hand-raising photos add no RQ2 evidence. `scripts/check_rules.py` checks this rule.

---

## 2. Scope

| | Item |
|---|---|
| **In (must do)** | Single photos · ~4–6 behaviours (D1) × the settings in D2 · Qwen2.5-VL-3B · zero-shot, few-shot, LoRA (normal training) · YOLO + flag table baseline · text-switch test (RQ2) · false-flag rate on exam photos (RQ3) · SCB + Roboflow exam_cheating · 2 stakeholder rounds · annotated-photo output |
| **Optional (only if week 2 ends early)** | Rule-contrastive LoRA (run B; only if RQ2 shows run A ignores the text, see Section6) · third test dataset (D5) · held-out setting · reworded-setting test · QLoRA · face-blur test · describe-then-reason prompt · second training seed · video-frame demo |

**Rule of thumb.** If a task does not answer RQ1–RQ3 or serve a rubric row, it is out. Ask the team before you add scope.

---

## 3. Research questions

| RQ | Question | Experiment | What each result would mean |
|---|---|---|---|
| **RQ1** | Does LoRA improve flagging over prompting alone, in both settings? | zero-shot vs few-shot vs LoRA; score lesson and exam settings separately | LoRA higher → training teaches cues that prompts cannot. Similar → prompting is enough. |
| **RQ2** | Does the model use the text, or guess the setting from the photo? | **Text-switch test:** same photo, lesson text vs exam text. Do the answers change where the flag table says they should? Before and after LoRA. | Answers change → the model reads the text. No change → it uses the photo style (SCB = classrooms, Roboflow = exams). If LoRA makes this worse, try run B (Section6). |
| **RQ3** | Is it safe enough for exams? | False-flag rate on exam photos with exam text: trained Qwen vs YOLO + flag table | Lower than YOLO → the VLM is safer for exams. Higher → state it as a limit for real use. |

**Headline claim, honest form:** "We test whether one VLM can serve lecturers and invigilators by text alone, whether LoRA helps, and whether it keeps using the text." Never claim more than the numbers show.

---

## 4. Model

**Qwen2.5-VL-3B-Instruct**, fine-tuned with LoRA.

How one question flows through the model:

| Step | Input | Part | What happens | Trained? |
|---|---|---|---|---|
| 1 | Photo | **Vision encoder** | Turns the image into image tokens | ❄️ frozen |
| 2 | Image tokens | **Projector** | Maps image tokens into the language model's space | ❄️ frozen |
| 3 | Setting + question (text) | **Text tokenizer** | Turns the words into text tokens | — |
| 4 | Image + text tokens | **Qwen2.5 language model** | Image and text tokens attend to each other in every layer (joint fusion) and write the answer | ❄️ frozen, **plus LoRA adapters 🔥 (the only trained part)** |
| 5 | Model text | **JSON answer** | `flagged` list (box + behaviour) and a `reason` | — |
| 6 | JSON | **Display script** | Draws boxes and a one-line summary for the reviewer | — (not part of the model) |

| Part | Detail | Why |
|---|---|---|
| Base model | Qwen2.5-VL-3B-Instruct | Fits one A40 (46 GB); has built-in box grounding; already verified in our earlier pipeline |
| Trained part | LoRA, rank 8, on language-model linear layers | Small and fast; the vision side already sees well |
| Frozen | Vision encoder, projector | Keeps training small; RQ2 is about text use, not vision |
| Fusion type | Joint: image and text tokens attend to each other in every layer | Contrast with YOLO (late fusion) |
| Input | photo (image) + setting and question (text) | 2 modalities, as the rubric requires |
| Output | JSON: `flagged` list of `{bbox_2d, behaviour}` + `reason` | A script can score it exactly |
| Box coordinates | Pixels of the image **after Qwen's resize** | That is how Qwen2.5-VL grounds boxes. `build.max_pixels` must match in training and inference. |
| Decoding | Greedy (temperature 0) | Repeatable results |
| Licence | Qwen Research License (non-commercial) | Fine for coursework |

---

## 5. Datasets

### Sources

| Dataset | Role | Behaviours it labels | Licence | Status (17 Oct) |
|---|---|---|---|---|
| **SCB, Hugging Face release** | lesson photos: train, val, test | discuss, TurnHead, BowHead, hand-raising, read, write | academic, non-commercial | ✅ downloaded and checked |
| **Roboflow [exam_cheating](https://universe.roboflow.com/behavior-cheating/exam_cheating-keaor)** | exam photos: train, val, test | Use_phone, Look_around, Cheat_paper, Normal (v2: 772 photos, ~136 scenes) | CC BY 4.0 | ✅ checked 17 Oct (Task 2a). v2 has no augmentation; photos stretched to 640×640 |
| **DIPSER** | third test set only (D5) | none of ours: attention/emotion labels | CC BY 4.0 | not started; would need 100–200 photos labelled by hand |

SCB classes `talk`, `using the phone` and `using the computer` appear in the SCB paper but are not in the public release.

### Behaviours: what we can label

| Behaviour | Source | Label quality (20-photo check) | Application |
|---|---|---|---|
| Talking (`discuss`) | SCB | 20/20 | both |
| Turning head (`TurnHead`) | SCB | 16/20 ⚠️ weakest | both |
| ~~Head down (`BowHead`)~~ | SCB | 19/20 | dropped (D1): often note-taking |
| Hand-raising | SCB | 19/20 | attention |
| Reading / writing | SCB | not checked | both |
| Using phone | Roboflow | correct ⚠️ box is on the **phone**; most photos are **not exam scenes** | invigilation |
| Looking around | Roboflow | 18/20 | invigilation |
| Reading notes (`Cheat_paper`) | Roboflow | 18/20 | invigilation |
| Normal (not flagged) | Roboflow | 20/20; **every student in a Roboflow photo is boxed** | gives "do not flag" examples; makes RQ3 false flags measurable |

Box counts in the SCB release: discuss 5,392 · TurnHead 11,156 · BowHead 4,962 · hand-raising 13,453 · read 24,078 · write 9,841.

### Known data problems, and how we handle them

| Problem | Handling |
|---|---|
| Each dataset or subset labels only its own classes | Frames shared by several SCB subsets are merged (union of labels, #32). Scoring ignores unlabelled behaviours, and **each question lists only the behaviours labelled for its photo (#30)**, so no training answer says "no one" about an unlabelled student |
| A few videos dominate; hand-raising was 57–63% of training (answer always "no one" in lecture and exam) | Cap per source video: train 20 / val 10 / test 30 (#31), and at most 500 hand-raising-only training photos (#33) |
| Every Discuss photo shows talking, so with its list ("talking") the answer never varies | Its behaviour-level score is trivial: report results per subset; RQ2 evidence comes from BowTurnHead and Roboflow |
| So talking is never scored on Roboflow photos, and phone is never scored on SCB photos | State it. RQ3 false flags cover only the behaviours Roboflow labels. |
| Every Discuss photo contains discussion: no "no one talking" examples | Open: accept as a limitation, or check ~50 other SCB photos by hand and use them as "no one" examples |
| TurnHead labels ~80% correct | Keep; state as a limitation |
| Frames from one video look almost the same; one video appears in several SCB subsets | Split by video **across subsets** (`split.group_regex`); `--check-leakage` finds near-copies |
| Roboflow: 772 photos are only ~136 scenes (copies, a 252-photo burst session, video frames); its own split leaks | Split by scene (#24, D6). Huge scenes → train only, thinned. Test = ~60 Roboflow photos from ~20 scenes: report exam per-behaviour scores as indicative |
| Roboflow: most Use_phone photos are not exam scenes | Keep them for training; score RQ3 only on scenes tagged `exam = yes` (#25, `scripts/scene_sheet.py`) |
| Roboflow: Use_phone boxes the phone, not the student | `box_recall` leaves out using_phone (#26) |
| Roboflow photos stretched to 640×640; mixed web and phone sources | State as a limitation |
| Two photo styles (classroom vs exam hall) | This is exactly what RQ2 tests: does the model use the text or the photo style? |
| SCB includes school-age students and Chinese classrooms | State as a limitation |

Download commands: `data/README.md`. Data is never committed.

---

## 6. Methods and baselines

All methods answer the **same test questions** and the **same script** scores them.

| Method | Reads the setting text? | Why it is here | How to run |
|---|---|---|---|
| **YOLO + flag table** | No; the table is code | The usual approach (late fusion). Answers "why not a detector + if-statement?" Reference for RQ3. | `export_yolo.py` + `yolo_rules_baseline.py` (one YOLOv8s on our train photos; D4, #40). Step by step: [docs/yolo_track.md](docs/yolo_track.md) |
| **Qwen zero-shot** | Yes | Starting point for RQ1, and the "before" for RQ2 | `run_vlm.py` |
| **Qwen few-shot** (2 examples) | Yes | Best result without training from examples (RQ1) | `run_vlm.py --few-shot 2` |
| **Qwen zero-shot + rules in prompt** | Yes, plus the flag-table row for its setting | Fair RQ1: does training beat simply **telling** the model the rules? (#35). Its participation score is not "unseen" (the rules are stated): report it separately | `run_vlm.py --rules-in-prompt` |
| **Qwen + LoRA, normal** (run A) | Yes | Our trained model: each training photo with one random setting | `lora_single.yaml` |
| *Optional:* **Qwen + LoRA, rule-contrastive** (run B) | Yes | A fix if run A ignores the text: each training photo with every setting | `lora_contrastive.yaml` |
| *Optional:* QLoRA | Yes | GPU memory saved vs accuracy lost | add `quantization_bit: 4` to a LoRA config |

**When to train run B.** Decide with the **validation** file, not the test file. After run A, run the text-switch check on val. Train run B only if run A's answers change with the text less often than zero-shot's do. If you decide after looking at test scores, the test scores are no longer fair.

**Fair comparison of runs A and B.** Run B has 2× the examples, so run A uses 2× the epochs. Both see each photo equally often. Only the setting pairing differs.

---

## 7. Pipeline step by step

Every script reads `configs/default.yaml`. Change settings there, never in code. Commands assume you are in the repo folder with the `classroom` env active.

### Prepare the data (steps 0–4)

#### Step 0. Check each dataset
- **You do:** run `python scripts/inspect_scb.py --root data/raw/scb`, then `scripts/preview_class.py` for each class (20 photos with the boxes drawn).
- **Why:** to know which classes exist, how many boxes they have, and whether the boxes are right.
- **Makes:** `results/data_check/scb_inspect.md` (counts and sample file names) and preview photos.
- **Done when:** for each dataset you have counts and a "right out of 20" score per class. **SCB: done. Roboflow: done (17 Oct).**

#### Step 1. Filter
- **You do:** set `scb_class_map` in `default.yaml` (dataset class name → our behaviour name), then run `python scripts/filter_scb.py`.
- **Why:** keeps only the boxes of the behaviours we use, under our names.
- **Makes:** `data/processed/manifest.jsonl`: one line per photo, with its size, dataset, the behaviours that dataset labels, and the kept boxes.
- **Used by:** steps 3 and 4.
- **Also:** merges frames shared by several SCB subsets into one photo with all their labels (#32).
- **Waits for:** D1. Every behaviour in `rules.csv` must be in the class map.
- **Done when:** the script prints no `WARNING: no boxes`.

#### Step 2. Flag table and settings
- **You do:** write the D3 table into `configs/rules.csv` (one column per setting). Write one sentence and 2 reworded versions per setting under `contexts:` in `default.yaml`. List the settings in `train_contexts` (and in `heldout_contexts` if you hold one out).
- **Why:** step 4 uses these to make the answer keys and the question text.
- **Note:** both files still hold the **old design** (computer lab, using computer). They must change before step 4. Check that the question text in `default.yaml` (`question:`) suits exam settings too; it says "this lesson".
- **Waits for:** D2, D3.
- **Done when:** `python scripts/check_rules.py` prints "All checks passed" (✅ 17 Oct). **The table is frozen.** Dr Lim reviews example outputs in round 2. If the table changes later, redo step 4 and re-run every baseline.

#### Step 3. Split the photos
- **You do:** `python scripts/split_data.py --check-leakage`
- **Why:** puts whole videos (SCB) or whole scenes (Roboflow, #24) into train (70%), val (15%) or test (15%). Frames of one video look almost the same, so they must stay in one pile, or the test score will be too high. The script prints the scene groups and the pile sizes per dataset.
- **Makes:** `data/processed/splits.json` and `data/processed/leakage_report.json` (test photos that look almost the same as a training photo).
- **Waits for:** step 1. (D6 is built in: `scene_subsets` in `default.yaml`.)
- **Done when:** you have looked at the leakage report and each dataset reaches all three piles.

#### Step 4. Make the question files
- **You do:** `python scripts/build_questions.py`
- **Why:** the model cannot learn from boxes; it needs question + answer pairs (Section1b).
- **Makes** (in `data/processed/questions/`):

  | File | Contents | Used by |
  |---|---|---|
  | `train_single.jsonl` | each training photo once, with one random setting | step 6, run A |
  | `train_contrastive.jsonl` | each training photo with every training setting | step 6, run B (optional) |
  | `val.jsonl` | val photos × training settings | steps 5–6: choose prompts and checkpoints |
  | `test.jsonl` | test photos × every setting | step 7, once per method; includes the text-switch pairs |
  | `test_reworded.jsonl` | test questions with reworded setting text | optional reworded-setting test |

- **Waits for:** steps 2 and 3.
- **Done when:** the script prints the counts; the share of "someone flagged" answers per setting is in `docs/experiment_log.md`; and 5 random questions are checked by eye (does the answer key match the photo?).

### Baselines (step 5)

#### Step 5. Score the methods that need no training
- **You do:** `run_vlm.py` on `val.jsonl` (zero-shot, then `--few-shot 2`), `yolo_rules_baseline.py`, then `evaluate.py` for each.
- **Why:** these are the scores that training must beat (RQ1), and the "before" for RQ2.
- **First time on the GPU:** add `--limit 20` and read the answers. They must be valid JSON.
- **Waits for:** step 4, GPU env, D4 (YOLO model).
- **Done when:** the val scores are in `results/summary.csv` and the few-shot choice is in the experiment log.

### Train (step 6)

#### Step 6. Train LoRA
- **You do:** `python scripts/export_llamafactory.py`, then `llamafactory-cli train configs/llamafactory/lora_single.yaml` (run A).
- **Why:** to teach the model the classroom and exam cues (RQ1).
- **Makes:** adapter checkpoints in `outputs/lora_single/` (not in git).
- **Choose the checkpoint** by val `flag_f1` (`run_vlm.py --adapter ...` on val), not by training loss.
- **Then:** the run B decision (Section6), on val.
- **Done when:** the chosen checkpoint, peak GPU memory and training time are in the experiment log.

### Compare (step 7)

#### Step 7. Final test
- **Before you start:** `evaluate.py` cannot yet report two numbers that Section3 needs (Section8, "to add"). Add them first.
- **You do:** run each final method **once** on `test.jsonl`, then `evaluate.py`.
- **Makes:** `results/metrics/*.json` and new rows in `results/summary.csv`.
- **Done when:** the results table (Section8) is filled and ~50 mistakes are sorted by type (missed student, wrong behaviour, ignored the text, wrong box).

### Report (step 8)

#### Step 8. Make the pictures and get feedback
- **You do:** `python scripts/visualize.py --questions ... --predictions ... --n 10`
- **Makes:** annotated photos with a one-line summary ("may need a check", never "cheating") in `results/figures/`.
- **Used by:** stakeholder round 2, the slides, the proposal.
- **Done when:** Dr Lim's round-2 feedback is in `docs/stakeholder_notes.md`, with what we changed.

---

## 8. Evaluation

### Metrics (`scripts/evaluate.py`)

| Metric | What it measures | Used for | Status |
|---|---|---|---|
| **`known_f1` (main)** | F1 over (question, behaviour) pairs, trained settings only (lecture, closed-book exam) | all RQs | ✅ |
| `ctx_<setting>_f1` | The same, per setting | RQ1 | ✅ |
| `flag_f1` | Every question in the file, held-out setting included | reference | ✅ |
| `flip_acc` | Trained settings: share of photos where every setting's answer is exactly right | RQ2 | ✅ |
| **`switch_rate`** | Photos whose correct answer changes between lecture and exam: share where the model's answer changes too | RQ2 (main) | ✅ 18 Oct |
| `false_switch` | Photos whose correct answer does not change: share where the model's answer changes anyway | RQ2 control | ✅ 18 Oct |
| **`rq3_precision`, `rq3_false_flag_rate`** | Roboflow photos tagged `exam = yes`, closed-book exam setting: share of flags that are right; share of photos with at least one wrong flag | RQ3 (#25) | ✅ 18 Oct |
| `by_behaviour`, `by_subset`, `by_dataset` | The same scores per behaviour, subset and dataset, per setting (in `results/metrics/*.json`; behaviour and dataset tables also printed) | talking is a free point; participation per row (#29) | ✅ 18 Oct |
| `unseen_f1`, `wording_para<n>_f1` | Held-out setting; reworded setting text | optional tests | ✅ |
| `box_recall` (secondary) | Flagged students found at IoU ≥ 0.5, `using_phone` left out (#26) | localisation | ✅ |
| `valid_json` | Share of answers that parse | invalid answers count as "no one flagged" | ✅ |

**Why F1, not accuracy.** Most checks are "not flagged". A model that never flags anyone would get high accuracy and zero F1.

### Protocol
1. Choose prompts, checkpoints and the run B decision on **val** only.
2. Run the **test** file once per final method. Do not tune after you see test scores.
3. One scorer (`evaluate.py`) for every method.
4. Every reported number comes from `evaluate.py` output (`results/summary.csv` or `results/metrics/*.json`), with `--note` saying the checkpoint or seed.

### Results table: final test (test, 755 photos / 2,265 questions; scored 20 Oct)

Main F1 = `known_f1` (lecture + exam, the trained settings). Participation = never trained. Switch rate (false switch) on images whose answer should change between lecture and exam. Exam false-flag rate = exam-tagged Roboflow photos with at least one wrong flag (41 photos: indicative only).

| Method | Main F1 | F1 lecture / exam | **F1 participation** | Switch rate (false) | `flip_acc` | Exam false-flag rate (precision) | Box recall | Valid JSON |
|---|---|---|---|---|---|---|---|---|
| YOLO + flag table | *pending* | | | n/a (no text) | | | | |
| Qwen zero-shot | 0.10 | 0.04 / 0.16 | 0.80 | 0.65 (0.20) | 0.10 | 5% (0.89) | 0.09 | 99.9% |
| Qwen few-shot (2) | 0.06 | 0.05 / 0.06 | 0.84 | 0.14 (0.18) | 0.21 | 0% (barely flags) | 0.06 | 98.5% |
| Qwen zero-shot + rules in prompt | 0.40 | 0.29 / 0.44 | 0.89 | 0.40 (0.01) | 0.77 | 5% (0.88) | 0.09 | 100% |
| **Qwen + LoRA (final checkpoint)** | **0.80** | **0.87 / 0.79** | **0.12** | **0.84 (0.04)** | **0.89** | 17% (0.81) | 0.09 | 98.4% |
| Qwen + LoRA, reworded setting sentences (4,530 q.) | 0.45 | 0.78 / 0.34 | 0.10 | 0.05 (0.01) | 0.74 | 0% (1.00) | 0.06 | 98.3% |

**Findings (test)**
- **RQ1:** fine-tuning beats every prompting baseline on the trained settings (0.80 vs 0.40 for rules-in-prompt).
- **RQ2:** with the exact training sentences, LoRA switches its answer correctly 84% of the time (false switch 4%). With **reworded** sentences it almost stops switching (0.05): it learned the training sentences, not their meaning (exam F1 0.79 → 0.34).
- **Participation (never trained):** LoRA 0.12 vs 0.80–0.89 for the untrained model. LoRA never points out hand-raising (0 / 489; it was always "do not point out" in training) but finds talking (0.89). The untrained model is the reverse: hand-raising 0.93, talking 0.00 in every setting.
- **Overall:** fine-tuning specialises: it learns the trained rules well but loses flexibility to follow new or reworded instructions. Future work: train with several wordings per setting and a few participation examples.
- **RQ3:** 17% of exam photos get at least one wrong flag (val: 40%); 41 photos only.
- Boxes remain weak (box recall 0.09).

### Validation results so far (val, 642 questions; NOT the final numbers)

Final wording v3 (decisions #37, #38). F1 = behaviour-level, trained settings. RQ3 = 58 exam-tagged Roboflow photos, closed-book exam setting (small: ±4 photos ≈ ±7 points).

| Method (v3 wording) | F1 lecture | F1 exam | Main F1 (`known_f1`) | Switch rate (false switch) | RQ3 precision / false-flag rate | Valid JSON |
|---|---|---|---|---|---|---|
| Qwen zero-shot | 0.05 | 0.14 | 0.09 | 0.70 (0.25) | 0.45 / 38% | |
| Qwen few-shot (2) | | | 0.06 | 0.10 | 1.00 / 0% (barely flags) | |
| Qwen zero-shot + rules in prompt | 0.37 | 0.30 | 0.32 | 0.43 (0.08) | 0.38 / 38% | |
| **Qwen + LoRA run A** (420 steps, 1024-token limit) | **0.77** | **0.76** | **0.76** | **0.90 (0.07)** | 0.57 / 40% | 98.4% |
| *LoRA run A, v2 wording (512-token limit)* | 0.72 | 0.74 | 0.74 | 0.88 (0.10) | 0.62 / 33% | 96.7% |

- **RQ1 (val):** LoRA 0.76 vs best prompting 0.32 (same wording). **RQ2 (val):** LoRA switches correctly 90% of the time, false switch 7%: run B not needed (#22 rule).
- **Prompt sensitivity (val, `known_f1`):** zero-shot v1 0.31 → v2 0.13 → v3 0.09; rules-in-prompt v2 0.38 → v3 0.32. Prompting swings with wording; LoRA does not need the wording to be right.
- **LoRA weak spots (val):** reading notes 2/18 found; turning head in exams 24 false flags; 40% of exam photos get ≥ 1 false flag; box recall 0.22; 9/642 answers hit the 1024-token limit (repeating boxes).
- v2 → v3 LoRA differs in wording AND token limit (#39): not a clean comparison, and not needed (wording frozen).

---

## 9. Lecture topics applied

| Topic | Week | Where | What it tests |
|---|---|---|---|
| Instruction tuning | 4 | step 4: photo + setting + question → JSON | teaches the task and an exact output format |
| LoRA | 4 | step 6 | does cheap fine-tuning help (RQ1), and does it keep text use (RQ2)? |
| QLoRA (optional) | 4 | step 6 | memory saved vs accuracy lost |
| Prompting: few-shot | 6 | step 5 | how far prompts go without training |
| Fusion: late vs joint | 5, 7 | YOLO + table vs Qwen | detector + code vs a VLM that reads the setting text |
| Trust and evaluation | 8 | step 7: false-flag rate, val/test rules, F1 | is it safe for exams (RQ3)? are scores fair? |

---

## 10. Timeline and dependencies

Put owner initials in the brackets. Tick the boxes when done and commit. **Submission deadline: [fill in]** · **Presentation: [fill in]**

**Order in one line:** Task 2a → Saturday decisions (D1–D3, D6) → config + Sunday email → steps 1–4 → baselines (step 5) → LoRA (step 6) → final test (step 7) → report (step 8). GPU set-up, YOLO training (after D4) and `evaluate.py` additions can run alongside.

### Sat 17 Oct

- [x] **Task 2a: check Roboflow exam_cheating** [Mambo]. *Needs: nothing.* **Done 17 Oct: passes with limits** (`decisions.md` #20, #24–#26).
  Download it in YOLOv8 format, from a version **without augmentation** if there is one (`data/README.md`), then run step 0 on it. Record: box counts, "right out of 20" for Use_phone, Look_around and Cheat_paper, whether photos are wide exam halls or close-ups, and whether the export holds edited copies of the same photo (D6).
  *Done when:* the results are posted to the group.
- [x] **Tag Roboflow scenes**: done 17 Oct (119 exam scenes, 24 not; phone rows being rechecked).
- [ ] **Team meeting** [everyone]. *Needs: Task 2a results.* **D1–D3 drafted by Mambo 17 Oct: the team confirms.**
  Confirm approach A (#22). Decide D1, D2, D3 (draft) and D6. Ideally D4 and D5 too. Check the D3 draft against the RQ2 constraint (Section1b).
  *Done when:* each decision is a row in `docs/decisions.md`.
- [x] **GPU environment**: done 18 Oct. Two environments: `classroom` (data, inference, scoring, YOLO) and `/projects/IE4422/envs/lf` (LLaMA-Factory training only; it needs older transformers/peft). Timing in #34.

### Sun 18 Oct

- [x] **Update the config**: done 17 Oct (class map, `rules.csv`, settings, neutral question).
- [ ] **Stakeholder email** [ ]. *Needs: the D3 draft.* Put the new flag table in the email; send it to Dr Lim.
  *Done when:* sent; the date is in `docs/stakeholder_notes.md`.
- [x] **Steps 1–4** [Mambo]. **Done 18 Oct** (with caps #31, #33). *Needs: the config update (done).*
- [x] **20-question GPU test** [Mambo]. **Done 18 Oct** (#34). *Needs: GPU env and `val.jsonl`.* `run_vlm.py --limit 20`; read the answers. **Write down seconds per question**: it sets the real cost of every test run.
- [x] **Training time check** [Mambo]. **Done 18 Oct** (#34). *Needs: GPU env.* Run LoRA run A for ~50 steps and time it. Use the speed to set `split.video_cap` (#31), then redo steps 3–4 before any baseline.

### Mon 19 – Thu 22 Oct

- [x] Zero-shot, few-shot and rules-in-prompt on val; scored (step 5) [Mambo]. **Done 18 Oct**, 3 wordings (#35–#38).
- [ ] YOLO: train one model on our training photos (if D4 agrees); YOLO + table on val [ ].
- [x] Add precision per setting/dataset and switch rate to `evaluate.py`, with a test (Section8). **Done 18 Oct**.
- [ ] Dr Lim's reply → freeze the flag table. If it changes: redo step 4 and re-run the baselines [ ].
- [ ] Cited figures for the problem statement (lecture size; invigilator-to-candidate ratio) [ ].

### Week 2 (23–29 Oct): training and final test

**Milestone:** every method scored on test; stakeholder round 2 done.
- [ ] LoRA run A; choose the checkpoint on val (step 6) [Mambo]. **Run A trained 18 Oct (v3, val known F1 0.76); checkpoint choice (200/400/420) left.**
- [x] Text-switch check on val → run B decision (Section6). **18 Oct: switch rate 0.90, false switch 0.07 → run B not needed.**
- [ ] Final test for every method (step 7) [ ].
- [ ] `visualize.py` examples; stakeholder round 2 (step 8) [ ].

### Week 3 (30 Oct – 5 Nov): results and write-up

**Milestone:** proposal submitted; viva rehearsed.
- [ ] Results table and error analysis (~50 mistakes by type) [ ].
- [ ] Apply round-2 feedback; proposal Section 8 [ ].
- [ ] Optional extras, only if week 2 ended early (Section2) [ ].
- [ ] Final proposal (≤ 2,000 words); verify every reference [ ].
- [ ] Slides; viva rehearsal: every member answers questions [ ].
- [ ] Peer evaluation on NTULearn (compulsory, 5% penalty) [everyone].

---

## 11. How to…

**Run and record an experiment**
1. `git pull`; note the commit hash.
2. Run the method (`run_vlm.py` / `yolo_rules_baseline.py`) → `results/predictions/<method>_<split>.jsonl`.
3. `python scripts/evaluate.py --method <method> --questions <file> --predictions <file> --note "<checkpoint, seed>"`
4. Add an entry to `docs/experiment_log.md`: command, commit, result, **what you observed**.
5. Commit `results/summary.csv`, `results/metrics/` and the log.

**Change the flag table** (only before it is frozen, or after a stakeholder request)
1. Edit `configs/rules.csv`.
2. Re-run `build_questions.py`. Earlier results are no longer comparable: re-score them.
3. Log it in `docs/decisions.md` with the reason.

**Add or change a behaviour**
1. Add a row to `rules.csv`; map the dataset class in `scb_class_map`.
2. Re-run steps 1, 3 and 4. Log the decision.

**Add a setting**
1. Add a column to `rules.csv` and an entry under `contexts` in `default.yaml` (one sentence + 2 reworded versions).
2. To test it as unseen, add it to `heldout_contexts`, not `train_contexts`.

**Add another dataset**
1. If it is YOLO format with a `.yaml` listing `names` and `images/` + `labels/` folders: put it under `data/raw/scb/<name>/`, map its class names in `scb_class_map`, and re-run steps 1, 3 and 4.
2. Check how its file names show the source photo or video (step 3). Edited copies of one photo must stay in one pile.
3. Otherwise, write a small script that outputs the **same manifest format** as `filter_scb.py` (image, width, height, subset, labelled, boxes).

---

## 12. Rules that keep results valid

| Don't | Because |
|---|---|
| Tune anything on test scores (this includes the run B decision) | the score is then optimistic and indefensible at the viva |
| Train on held-out-setting questions or reworded setting text | they are the unseen-setting and reworded tests |
| Change `max_pixels` between training and inference | boxes will be in the wrong pixel space |
| Compare runs with different `rules.csv` or question files | differences come from the data, not the method |
| Report numbers from a terminal or notebook | only `evaluate.py` output is traceable |
| Commit data, images, weights or tokens | licences and repo size |
| Write "cheating" or "not paying attention" in outputs | we see behaviour, not intent |
| Claim "first to…" without checking | examiners probe absolute claims |

---

## 13. Risks and fallbacks

| Risk | Sign | Fallback |
|---|---|---|
| Roboflow labels poor, or close-ups only | Task 2a: **passed 17 Oct** | Look for another exam dataset; else invigilation uses SCB behaviours only and phone moves to future work |
| Roboflow copies or scenes leak into test | leakage report after step 3 | Scene split is in place (#24). If the report still shows Roboflow near-copies across piles, `scene_hamming_threshold` is too low: raise it to 7 and re-split |
| RQ2 cannot be tested | No row of the D3 table changes between settings for one dataset | Change the settings (D2), e.g. use group discussion as the lesson setting |
| Too few boxes for a behaviour | filter WARNING, or < ~100 boxes | Drop or merge it; log the decision |
| `group_regex` cannot find source videos | split prints "almost one group per image" | Rely on `--check-leakage`; group near-copies by hand |
| VLM answers are not valid JSON | `valid_json` < 0.9 | Add one example to the prompt; check `max_new_tokens` |
| Box recall near 0 for every method | crowded rooms, small students | Keep `flag_f1` as the headline; report boxes as a limitation |
| GPU queue or memory problems | out-of-memory errors, long waits | Batch size 1 + more gradient accumulation; QLoRA; Colab as backup |
| LoRA run late | week 2 slipping | Drop optional items; run B only if clearly needed |

---

## Key files

| File | What it is |
|---|---|
| `configs/default.yaml` | every setting: paths, class map, setting sentences, split, model |
| `configs/rules.csv` | the flag table |
| `configs/llamafactory/*.yaml` | LoRA runs A and B |
| `data/README.md` | how to download each dataset |
| `docs/decisions.md` | what we decided and why (for the viva) |
| `docs/stakeholder_notes.md` | stakeholder contact, questions and changes made |
| `docs/experiment_log.md` | what was run and what we observed |
| `results/summary.csv`, `results/metrics/` | the only sources of reported numbers (written by `evaluate.py`) |
