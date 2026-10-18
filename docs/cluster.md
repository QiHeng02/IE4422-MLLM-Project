# Running the project on the NTU EEE GPU cluster

**Default: Mambo runs the jobs.** Mambo's project folder (`/projects/IE4422`) is private to his account, so it cannot be
shared. If you want to run things yourself, build **your own copy** with this guide: about 1–2 hours, mostly waiting
for downloads. You get the same code, data, model and results, under your own account.

## 0. What you need

- An EEE cluster account, and the NTU network (on campus or NTU VPN).
- About **30 GB of disk** that you own: model ~7.5 GB, two Python environments ~8–10 GB each, photos a few GB.
- The Roboflow zip, downloaded in your browser (step 5).

Never commit or push from the cluster: code changes go laptop → GitHub → `git pull` on the cluster.

## 1. Log in and choose your folder

```bash
ssh <your-username>@10.97.216.128
quota -s 2>/dev/null; df -h ~        # how much space your home folder has
```

If your home folder has 30 GB free, use it. If not, ask the cluster admin for a project folder (that is how
Mambo got `/projects/IE4422`). Then tell every future session where it is:

```bash
echo 'export IE4422_HOME=$HOME/ie4422' >> ~/.bashrc     # or your project folder
source ~/.bashrc
mkdir -p $IE4422_HOME/{envs,hf_cache,data/scb}
```

You land on a **login node**: fine for installing, downloading, `git pull` and scoring. **No GPU work here.**

## 2. Get the code

```bash
git clone https://github.com/QiHeng02/IE4422-MLLM-Project.git $IE4422_HOME/IE4422-MLLM-Project
cd $IE4422_HOME/IE4422-MLLM-Project
mkdir -p data/raw logs
ln -s $IE4422_HOME/data/scb data/raw/scb
```

## 3. Python environments (two, on purpose)

`classroom` runs everything except training. `lf` runs LoRA training only: LLaMA-Factory needs older versions of
transformers/peft and would break `classroom` (decisions #34). Install the **exact** versions Mambo used, from the
lock files in the repo:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
PYV=$(head -1 requirements-lock-classroom.txt | grep -o "[0-9.]*" | head -1)   # first line: "# python 3.11"
conda create -p $IE4422_HOME/envs/classroom python=$PYV -y
conda activate $IE4422_HOME/envs/classroom
pip install -r requirements-lock-classroom.txt
pip install -e .

PYV=$(head -1 requirements-lock-lf.txt | grep -o "[0-9.]*" | head -1)
conda create -p $IE4422_HOME/envs/lf python=$PYV -y
conda activate $IE4422_HOME/envs/lf
pip install -r requirements-lock-lf.txt
```


## 4. Download the model (once)

```bash
conda activate $IE4422_HOME/envs/classroom
export HF_HOME=$IE4422_HOME/hf_cache
hf download Qwen/Qwen2.5-VL-3B-Instruct
```

## 5. Download the photos

Follow [data/README.md](../data/README.md), but put everything in **your** `$IE4422_HOME/data/scb`
(read `/projects/IE4422` there as `$IE4422_HOME`):

- **SCB:** 3 subsets, one `hf download` each, then unzip.
- **Roboflow exam_cheating v2:** download the YOLOv8 zip **in your browser** (the cluster is blocked by Cloudflare),
  `scp` it to `$IE4422_HOME/data/scb/exam_cheating/`, and unzip it there.

The exam / not-exam scene tags are already in the repo (`configs/scene_tags.csv`): do not redo them.

## 6. Build the processed data (steps 1–4)

Same code, same settings, same seed, so you get the same piles as Mambo:

```bash
cd $IE4422_HOME/IE4422-MLLM-Project
conda activate $IE4422_HOME/envs/classroom
export PYTHONPATH=$PWD/src
python scripts/check_rules.py
python scripts/filter_scb.py
python scripts/split_data.py --check-leakage
python scripts/build_questions.py
python scripts/export_llamafactory.py
```

**Check:** the photo counts must be **train 1,668 / val 321 / test 755**, and the questions
**val 642 / test 2,265**. If they differ, stop and tell Mambo: something in your download is different, and your
results would not be comparable.

## 7. Run GPU work as a batch job

A batch job keeps running after you disconnect. **Submit from your repo folder**: the job runs there, uses your
`$IE4422_HOME` environments and model, and writes the log to `logs/<job-name>-<job-id>.out`. Time limit: 6 hours.

```bash
# example: the untrained model on 20 val questions
sbatch --job-name=my_test jobs/run.sbatch python scripts/run_vlm.py \
  --questions data/processed/questions/val.jsonl \
  --out results/predictions/my_test_val.jsonl --limit 20

# training uses the other environment:
ENV=$IE4422_HOME/envs/lf sbatch --job-name=my_lora jobs/run.sbatch \
  llamafactory-cli train configs/llamafactory/lora_single.yaml output_dir=outputs/my_lora
```

Check on jobs:

```bash
squeue --me                                    # running / waiting jobs (empty = all finished)
sacct -u $USER --starttime today --format=JobID,JobName%15,State,Elapsed
tail -c 600 logs/my_test-*.out | tr '\r' '\n' | tail -3     # latest progress line
scancel <job-id>                               # stop a job
```

**Tell Mambo what you are running**, so two people do not run the same job.

## 8. Interactive GPU session (quick tests only)

```bash
srun --gres=gpu:a40:1 --time=1:00:00 --pty bash
```

Ask for nothing more (no `--cpus-per-task`, no `--mem`): the cluster allows 4 CPUs and 40 GB per A40 and rejects
larger requests. Type `exit` when done so the GPU is freed.

## 9. Score a run (login node)

```bash
python scripts/evaluate.py --method my_test \
  --questions data/processed/questions/val.jsonl \
  --predictions results/predictions/my_test_val.jsonl
```

## Common problems

| Problem | Fix |
|---|---|
| `ssh: Connection timed out` | Not on the NTU network: connect to the NTU VPN |
| `More processors requested than permitted` | Remove `--cpus-per-task` / `--mem` from `srun` |
| `Permission denied` on `/projects/IE4422` | That is Mambo's private folder: use your own `$IE4422_HOME` |
| Job cannot find the environment or model | `IE4422_HOME` not set: check `echo $IE4422_HOME`, then step 1 |
| Roboflow download is a 5 KB file | Cloudflare blocked the cluster: download in your browser (step 5) |
| Different photo counts in step 6 | Stop and tell Mambo before running anything |
| Job finished instantly | Read its log in `logs/`; usually a typo in a path |
| `ImportError: libGL.so.1` | `pip uninstall -y opencv-python && pip install opencv-python-headless` |
