# How we work in this repo

## Before every commit

- `pytest -q` passes.
- `git status` shows no data, images, weights, `.env` or tokens.
- If you changed data, metrics or training: add a line to `docs/decisions.md`.
- If you ran an experiment: add an entry to `docs/experiment_log.md`.

## Where things go

| You made… | Put it in |
|---|---|
| a reusable function | `src/ie4422/` (+ a test in `tests/`) |
| a pipeline step | `scripts/`, reading `configs/default.yaml` |
| a setting | `configs/default.yaml`, never hard-coded |
| a notebook for exploring | `notebooks/` (clear outputs before committing) |
| a result | `results/summary.csv` via `scripts/evaluate.py` |

## Using AI tools

Course policy: GenAI is allowed for language editing and implementation help, **not for idea generation**.
Research decisions (what to test, why) are made by the team and logged in `docs/decisions.md`.
Every member must be able to explain any code they commit at the viva.
