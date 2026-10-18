# Context-Aware Classroom Behaviour Analysis with VLMs

IE4422 (Multimodal LLMs) group project, NTU.

**One line:** one vision-language model looks at a classroom or exam-hall photo **and** reads one sentence that says the setting (e.g. "This is a lecture." or "This is a closed-book exam."). It flags the students whose visible behaviour does not fit that setting. Two users, one model: **lecturers** (class attention) and **invigilators** (exams).

We ask three questions: does LoRA fine-tuning beat prompting alone, does the model really use the setting text, and how often does it flag an innocent student in an exam? See [PROJECT_PLAN.md §3](PROJECT_PLAN.md#3-research-questions).

| | |
|---|---|
| Model | Qwen2.5-VL-3B-Instruct + LoRA |
| Input | photo (image) + setting and question (text) |
| Output | JSON: flagged behaviours, a box per student, a one-line reason |
| Data | SCB-Dataset (lesson photos, ready) + Roboflow exam_cheating (exam photos, being checked) |
| **Project guide** | [PROJECT_PLAN.md](PROJECT_PLAN.md): status, decisions, scope, pipeline, timeline |

## How the pipeline works

1. **Prepare the data.** Keep the behaviours we use, split the photos into training / validation / test piles, and write question files. Each question is one photo + one setting sentence; its answer key comes from boxes that people drew on the photo.
2. **Baselines.** Score the model with no training, the model with 2 examples in the prompt, and a YOLO detector + a lookup table.
3. **Train.** Fine-tune the model with LoRA on the training questions.
4. **Compare.** Every method answers the same test questions once; one script scores them all.
5. **Report.** Draw the flags on photos for the stakeholder and the slides.

What each step does, what it makes and when it is done: [PROJECT_PLAN.md §7](PROJECT_PLAN.md#7-pipeline-step-by-step).

## Set up

```bash
git clone https://github.com/QiHeng02/IE4422-MLLM-Project.git
cd IE4422-MLLM-Project
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest -q                                                # 8 tests, a few seconds, no GPU, no data needed
```

On the NTU GPU cluster, follow [docs/cluster.md](docs/cluster.md) instead. Data downloads: [data/README.md](data/README.md).

## Repo layout

```
configs/          default.yaml (all settings), rules.csv (flag table), llamafactory/ (LoRA runs)
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
- **Every number in the report comes from `evaluate.py` output** (`results/summary.csv`, `results/metrics/`). Do not copy numbers from a terminal.
- **Log every decision** that changes data, metrics or training in [docs/decisions.md](docs/decisions.md). The viva will ask "why".
- Before each commit, follow [CONTRIBUTING.md](CONTRIBUTING.md).

## Licences

- SCB-Dataset: academic and non-commercial use only ([Whiffe/SCB-dataset](https://github.com/Whiffe/SCB-dataset)).
- Roboflow exam_cheating, DIPSER: CC BY 4.0.
- Qwen2.5-VL-3B: Qwen Research License (non-commercial).

This repo is coursework for research use only.
