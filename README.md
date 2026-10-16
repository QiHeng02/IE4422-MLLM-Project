# Context-Aware Classroom Behaviour Analysis with VLMs

IE4422 (Multimodal LLMs) group project, NTU.

**One line:** a vision-language model reads a classroom or exam-hall photo **and** a rule written as text, then flags students whose visible behaviour breaks that rule. Two applications, one model: **class attention** (lecturers) and **invigilation** (exams). We test whether LoRA fine-tuning keeps the model reading the rule.

> Reading notes is cheating in a closed-book exam but fine in an open-book one. A fixed detector gives the action the same label in both. Our model gets the rule as input.

| | |
|---|---|
| Model | Qwen2.5-VL-3B-Instruct + LoRA |
| Input | classroom / exam photo (image) + rule and question (text) |
| Output | JSON: flagged behaviours, a box per student, a one-line reason |
| Data | SCB-Dataset (ready) + one exam-behaviour dataset for phone use (being checked); second test set TBC |
| **Project guide** (scope, model, data, baselines, pipeline, timeline) | [PROJECT_PLAN.md](PROJECT_PLAN.md) |

## Set up

```bash
git clone https://github.com/QiHeng02/IE4422-MLLM-Project.git
cd IE4422-MLLM-Project
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest -q                                                # 5 tests, ~1 s, no GPU, no data needed
```

On the GPU cluster, also `pip install -r requirements-gpu.txt`.

## Pipeline

Every script reads `configs/default.yaml`. Change settings there, not in code. Step numbers match the proposal.

| Step | Command | Output | Needs |
|---|---|---|---|
| 0. Check data | `python scripts/inspect_scb.py --root data/raw/scb` | class counts, sample file names | SCB download |
| 1. Filter | `python scripts/filter_scb.py` | `data/processed/manifest.jsonl` | filled `scb_class_map` |
| 2. Rule table | edit `configs/rules.csv` | — | stakeholder round 1 |
| 4a. Split | `python scripts/split_data.py --check-leakage` | `splits.json`, leakage report | |
| 3+4b. Questions | `python scripts/build_questions.py` | `questions/*.jsonl` | |
| 5. VLM baselines | `python scripts/run_vlm.py --questions .../val.jsonl --out results/predictions/zeroshot_val.jsonl` (add `--few-shot 2`) | predictions | GPU |
| 5. YOLO baseline | `python scripts/yolo_rules_baseline.py --weights best.pt --questions ... --out ...` | predictions | YOLO weights |
| 6. LoRA | `python scripts/export_llamafactory.py` then `llamafactory-cli train configs/llamafactory/lora_contrastive.yaml` (and `lora_single.yaml`) | `outputs/lora_*` | GPU |
| 7. Evaluate | `python scripts/evaluate.py --method NAME --questions ... --predictions ...` | `results/metrics/`, `results/summary.csv` | |
| 8. Display | `python scripts/visualize.py --questions ... --predictions ... --n 10` | annotated PNGs | |

Status of each step and who owns it: see the to-do list in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Repo layout

```
configs/          default.yaml (all settings), rules.csv (rule table), llamafactory/ (LoRA runs)
scripts/          one script per pipeline step
src/ie4422/       shared code: boxes, rules, parsing (pip install -e .)
tests/            unit tests + an end-to-end test on a fake dataset
docs/             decision log, stakeholder notes, experiment log
data/             NOT in git: raw/ (downloads), processed/ (generated)
results/          summary.csv + metrics/ in git; predictions/ and figures/ not
outputs/          NOT in git: LoRA checkpoints
```

## Rules for the repo

- **Never commit data, images, weights or tokens.** `.gitignore` blocks the usual paths; check `git status` before you commit.
- **Every number in the report comes from `results/summary.csv`.** Do not copy numbers from a terminal.
- **Log every decision** that changes data, metrics or training in [docs/decisions.md](docs/decisions.md). The viva will ask "why".
- Work on a branch, open a pull request, one reviewer. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licences

- SCB-Dataset: academic and non-commercial use only ([Whiffe/SCB-dataset](https://github.com/Whiffe/SCB-dataset)).
- Roboflow exam_cheating, DIPSER: CC BY 4.0.
- Qwen2.5-VL-3B: Qwen Research License (non-commercial).

This repo is coursework for research use only.
