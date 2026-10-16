# Project guide

The team's reference for this project. Read sections 1–3 once to understand the project. Come back to sections 4–12 whenever you work on it.

| Section | Read it when you want to know… |
|---|---|
| [1. The project in one page](#1-the-project-in-one-page) | what we are doing and why |
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

## 0. Open blocker: check the behaviour classes first

Our rule table uses 4 behaviours: **talking, turning head, using phone, using computer**.

**What we know (checked 16 Oct).** The SCB paper (arXiv:2304.02488, latest version) lists 20 classes, including `talk`, `turn head`, `using the phone` and `using the computer`. But the public subsets on [Hugging Face](https://huggingface.co/datasets/wintonYF/SCB-Dataset) only show:

| Public subset | Classes (verified from its yaml) | Useful for us |
|---|---|---|
| SCB5-Discuss | `discuss` | talking (proxy) |
| SCB_BowTurnHead | `BowHead`, `TurnHead` | turning head ⚠️ the paper says training on these was "unsatisfactory": expect noisy labels |
| SCB5-Handrise-Read-write | `hand-raising`, `read`, `write` | no |
| SCB5_Teacher_Behavior | teacher classes | no |
| SCB_LLM_202506 | activity-level zips (VLM data), no class list | unclear |
| **YOLO.zip (5.3 GB)** | **not listed; most likely place for `talk`, `using the phone`, `using the computer`** | check first |

**Steps (owner: data workstream):**

1. On the NTU cluster (campus Wi-Fi; the VPN is slow), download `YOLO.zip` and the two small subsets:
   ```bash
   D=/projects/IE4422/data/scb
   hf download wintonYF/SCB-Dataset YOLO.zip --repo-type dataset --local-dir $D
   hf download wintonYF/SCB-Dataset --repo-type dataset --include "SCB5-Discuss-2024-9-17/*" --local-dir $D
   hf download wintonYF/SCB-Dataset --repo-type dataset --include "SCB_BowTurnHead_20250509/*" --local-dir $D
   unzip -l $D/YOLO.zip | grep -iE "\.ya?ml$"    # list first, unzip after
   ```
2. Run `python scripts/inspect_scb.py --root <unzipped folder>` on each.
3. Team decides, logged in `docs/decisions.md` (entry 12):
   - **A.** YOLO.zip has phone, computer and talk with enough boxes → keep the 4 behaviours.
   - **B.** It does not → use what SCB labels: `discuss`, `TurnHead`, `BowHead`, `hand-raising`. Rule table and proposal change (e.g. head down: OK in exam, flag in lecture).
   - **C.** Mix: SCB for discuss / turn head + another dataset for phone / computer. Candidates: [SAV](https://github.com/Ritatanz/SAV) (15 classes incl. "talking with others", "looking around"; no phone or computer; check licence), or a COCO-pretrained detector for phone and laptop (boxes only, no human labels).

The code does not care which: only `configs/rules.csv` and `scb_class_map` change. Delete this section once decided.

---

## 1. The project in one page

**Users.** Lecturers and TAs in large classes. They cannot watch every student while teaching.

**Problem.** Current tools detect fixed actions (talking, phone use) and give each one the same label in every lesson. But whether an action is OK depends on the lesson:

| Action | Group discussion | Closed-book exam |
|---|---|---|
| Talking | OK | flag |
| Turning head | OK | flag |

**Our idea.** Give a vision-language model (VLM) the photo **and** the lesson rule as text. The rule decides the answer, so the same photo has different correct answers under different rules.

```
photo  +  "The class is sitting a closed-book exam. Which students should be flagged?"
  →  {"flagged": [{"bbox_2d": [412,130,520,390], "behaviour": "talking"}], "reason": "..."}
```

**The research question.** When we fine-tune the VLM with LoRA, does it **keep reading the rule**, or does it learn to answer from the picture alone? We also test a fix: **rule-contrastive training**, where every training image appears with every rule and its different answer.

**Why this matters.** If the model reads the rule, a lecturer can add a new lesson type by writing one sentence. No new code, no retraining. If it ignores the rule, it is just an expensive detector.

**Any result is a finding.** The course marks what we try, observe and explain, not whether the method wins.

---

## 2. Scope

| | Item |
|---|---|
| **In (must do)** | Single photos · 4 behaviours × 3 lesson types · Qwen2.5-VL-3B · zero-shot, few-shot, LoRA (normal + rule-contrastive) · YOLO + rule table baseline · flip test, unseen-rule test, reworded rules · SCB + one second dataset · 2 stakeholder rounds · annotated-photo output |
| **Optional (only if week 2 ends early)** | QLoRA · face-blur test · describe-then-reason prompt · second training seed · video-frame demo |
| **Out** | Video or audio input · CLIP baseline · self-consistency · LoRA rank sweep · our own NTU photos · real-time deployment · measuring attention or emotion · identifying or tracking students |

**Rule of thumb.** If a task does not answer RQ1–RQ3 or serve a rubric row, it is out. Ask the team before adding scope.

---

## 3. Research questions and what counts as a result

| RQ | Question | Experiment | Possible findings |
|---|---|---|---|
| **RQ1** | Does LoRA beat prompting? | zero-shot vs few-shot vs LoRA on known rules (`known_f1`) | LoRA higher → training teaches classroom cues. Similar → prompting is enough for this task. |
| **RQ2** | Does LoRA keep the model reading the rule? Does rule-contrastive training help? | flip test, unseen-rule test, reworded rules | Normal LoRA drops on `unseen_f1` or `flip_acc` vs zero-shot → fine-tuning made it ignore text. Contrastive recovers it → our fix works. No drop → fine-tuning is safe here. |
| **RQ3** | When is a VLM worth it over a detector? Does it transfer? | YOLO + rule table vs Qwen; best model on DIPSER | YOLO wins known rules → detectors are better at finding objects. VLM wins new rules without code → its value is flexibility. DIPSER drop → domain shift. |

**Headline claim, honest form.** "We test whether LoRA fine-tuning keeps a VLM reading lesson rules, and whether rule-contrastive training protects it." Never claim more than the numbers show.

---

## 4. Model

**Qwen2.5-VL-3B-Instruct**, fine-tuned with LoRA.

```
Classroom photo ─→ Vision encoder (frozen) ─→ Projector (frozen) ─┐
                                                                  ├─→ Qwen2.5 language model ─→ JSON answer ─→ display script
Lesson rule + question ─→ Text tokenizer ─────────────────────────┘    (+ LoRA adapters: the only trained part)
```

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

| Dataset | Role | What it gives | Licence | Status |
|---|---|---|---|---|
| **SCB-Dataset** | train, val, main test | classroom photos + behaviour boxes (YOLO format), per subset | academic, non-commercial | ⚠️ classes to check (§0) |
| **DIPSER** | second test only | university classrooms (Spain); attention/emotion labels, not our behaviours | CC BY 4.0 | to request; we hand-label 100–200 frames |
| HCCB / Roboflow "Students in Lecture" | backup second test | lecture-hall photos | check | only if DIPSER fails |

**Known data problems, and how the code handles them:**

| Problem | Handling |
|---|---|
| Each SCB subset labels only its own classes (a phone user in the Discuss subset is unlabelled) | `filter_scb.py` records `labelled` per image; scoring ignores unlabelled behaviours |
| Frames from one video look almost identical | split by source video (`split.group_regex`); `--check-leakage` finds near-twins |
| A plain split can leave a whole subset (= a behaviour) out of test | split is stratified by subset |
| SCB includes school-age students and Chinese classrooms | state as a limitation; DIPSER measures transfer to adult classrooms |
| SCB photos are normal lessons, not real exams | state it: we test rule-following, not exam scenes |

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
- **Done when:** behaviour set decided (§0); `scb_class_map` and `split.group_regex` set.

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
- **Why:** a VLM needs question + answer pairs; splitting by video stops leakage; computer lab stays out of training for the unseen-rule test.
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
| `known_f1` / `unseen_f1` | `flag_f1` on trained lesson types / on computer lab only | RQ1 / RQ2 |
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

| Method | flag_f1 (SCB) | flip_acc | unseen_f1 | reworded F1 | flag_f1 (DIPSER) |
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

- [ ] Confirm topic with instructor; check no other group has it [ ]
- [ ] Assign an owner to every line [ ]
- [ ] §0 blocker: download SCB subsets, run `inspect_scb.py`, decide A/B/C [ ]
- [ ] Set `split.group_regex` from sample file names [ ]
- [ ] Draft `rules.csv`; stakeholder round 1; freeze; notes [ ]
- [ ] `filter_scb.py` → `split_data.py --check-leakage` → `build_questions.py` [ ]
- [ ] GPU env on the A40 (`requirements-gpu.txt`); `run_vlm.py --limit 20` [ ]
- [ ] Zero-shot on val; score it [ ]
- [ ] Find SCB-trained YOLO weights, or start YOLOv8n training [ ]
- [ ] Request DIPSER access; check licence [ ]
- [ ] Cited class-size figures for proposal Section 1 [ ]

### Week 2 (23–29 Oct): training and tests

**Milestone:** every method scored on test; stakeholder round 2 done.

- [ ] Few-shot on val [ ]
- [ ] YOLO + rule table on test [ ]
- [ ] Export data; LoRA run A (normal) [ ]
- [ ] LoRA run B (rule-contrastive) [ ]
- [ ] Pick checkpoints on val; run all methods on `test` and `test_reworded` [ ]
- [ ] Hand-label 100–200 DIPSER frames; DIPSER manifest [ ]
- [ ] `visualize.py` examples; stakeholder round 2; notes [ ]

### Week 3 (30 Oct – 5 Nov): results and write-up

**Milestone:** proposal submitted; viva rehearsed.

- [ ] Best model on DIPSER; final results table [ ]
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

**Add a second test dataset (DIPSER)**
1. Write a small script that outputs the **same manifest format** as `filter_scb.py` (image, width, height, subset, labelled, boxes).
2. Put all its images in test in a splits file; run `build_questions.py` with a config pointing at them.

---

## 12. Rules that keep results valid

| Don't | Because |
|---|---|
| Tune anything on test scores | the score is then optimistic and indefensible at the viva |
| Train on computer-lab questions or paraphrased wordings | they are the unseen-rule and reworded tests |
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
| SCB lacks phone / computer classes | `inspect_scb.py` shows none | §0 options B or C |
| Too few boxes for a behaviour | filter WARNING, or < ~100 boxes | drop or merge it; log the decision |
| `group_regex` cannot find source videos | split prints "almost one group per image" | rely on `--check-leakage`; group near-twins by hand |
| VLM answers are not valid JSON | `valid_json` < 0.9 | add one example to the prompt; check `max_new_tokens` |
| Box recall near 0 for every method | crowded rooms, small students | keep `flag_f1` as the headline; report boxes as a limitation |
| GPU queue or memory problems | OOM, long waits | batch size 1 + more gradient accumulation; QLoRA; Colab as backup |
| LoRA run late | week 2 slipping | drop optional items; train run B first (it is our method) |
| DIPSER unavailable or too slow to label | no access by mid week 2 | backup dataset (§5), or ~100 labelled frames |
| No YOLO weights for our classes | none in SCB release | COCO-pretrained YOLO for phone and laptop only; say so in the report |

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
