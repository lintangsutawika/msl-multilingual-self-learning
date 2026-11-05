# MLE training data: Kaggle tasks with a public and a private leaderboard

Builds RL training tasks for ML-engineering agents from **MLE-Dojo**, which provides real Kaggle competitions plus
DSBench's Kaggle-derived tasks, and packages them as Harbor tasks with a public/private leaderboard split.
This is training data only; the eval benchmark is `src/benchmarks/mlebench/`, and **every MLE-bench competition is
excluded**.

How a task works:
- During the run the agent calls `validate <file.csv>`. It scores the file on the **public** split, a fixed random
  30% of the test ids, and logs the score. These scores are the per-episode signal: an episode is the agent steps
  between two `validate` calls.
- The agent runs as the unprivileged user `agent`. `validate` reaches the root-only grader and public answers
  through one `sudo` rule and prints a signed `score=<s> sig=<hmac>` line (HMAC with the task's
  `environment/score_key`). The rollout forwarder only trusts signed scores (`SCORE_KEYS_DIR` = the tasks folder).
- After the run, the verifier grades `/workspace/predictions.csv` on the **private** split, the other 70%. The reward
  is the submission's position on the competition's real Kaggle private leaderboard, in [0, 1].

The agent always predicts every test row; each split only scores its own ids, as on Kaggle.

## Quick start (any Linux machine or cluster)

You need:
- **Python ≥ 3.11**, git and internet access;
- **for Kaggle competitions:** a Kaggle API key in `~/.kaggle/kaggle.json` (kaggle.com → Settings → API → Create New
  Token), and each competition's rules accepted on kaggle.com;
- **disk:** about 50 GB to download (Kaggle zips up to 5 GB each) plus 3 GB for DSBench; the unzipped data takes a
  few times that. `--max-gb` lowers the size cap.

From the repo root:

```bash
uv venv --python 3.11 .venv-mletrain                                  # or: python3.11 -m venv .venv-mletrain
uv pip install --python .venv-mletrain -r src/benchmarks/mle_train/requirements.txt
.venv-mletrain/bin/python src/benchmarks/mle_train/pipeline.py all --workdir ~/mle-train
```

`all` does everything:
1. clones MLE-Dojo at the pinned commit;
2. downloads the data;
3. imports the tasks, splits them and runs the quality checks;
4. builds the Harbor tasks and writes the training jsonl.

The output:

```
~/mle-train/MLE-Dojo/        MLE-Dojo checkout (pinned commit)
~/mle-train/raw/mledojo/     raw downloads (kaggle/_status.tsv says what downloaded)
~/mle-train/root/            every task in MLE-bench layout; index.tsv says which are usable and why not
~/mle-train/harbor/          Harbor tasks              -> HARBOR_TASKS_DIR
~/mle-train/train.jsonl      training prompt file      -> --prompt-data
```

At the end it prints the Kaggle rules pages for competitions Kaggle refused. Accept them on kaggle.com and **rerun
the same command**: downloads resume, finished competitions are skipped, and everything after is rebuilt.

- **No Kaggle account?** `--no-kaggle` builds only the DSBench tasks (about 58), which download from Hugging Face.
- **On a Slurm cluster:** `slurm/build.sbatch` wraps the same command. It also runs with plain `bash`:

  ```bash
  mkdir -p logs
  sbatch --export=ALL,WORKDIR=$HOME/mle-train,PYTHON=$PWD/.venv-mletrain/bin/python \
         [your account/partition flags] src/benchmarks/mle_train/slurm/build.sbatch
  ```

  On CMU Babel the flags are `-A teruko -p preempt -q preempt_cpu_qos`. The job can be requeued safely.
- **Check the tasks with Harbor:** `slurm/oracle_check.sbatch` runs the oracle agent, which submits the sample
  submission, on every task. It uses Harbor's Docker environment by default. Where there is no Docker (e.g.
  Singularity on Babel), set `HARBOR_ENV_ARGS`; the script header shows how.

```bash
TASKS=~/mle-train/harbor bash src/benchmarks/mle_train/slurm/oracle_check.sbatch
```

- **Run the unit tests** (no Kaggle, about 15 s):

  ```bash
  .venv-mletrain/bin/python -m pip install pytest
  .venv-mletrain/bin/python -m pytest src/benchmarks/mle_train/tests -q
  ```

Only `pipeline.py` and its folder are needed. The parent package isn't imported, so `mlebench`, Harbor and the
repo's own environment aren't required to build the data. Harbor is only needed to run the tasks.

### Step by step

`all` chains these subcommands; each one can also be run on its own:

```bash
P=src/benchmarks/mle_train/pipeline.py
python $P mledojo-download --dojo-repo DOJO --data-dir RAW --max-gb 5 --dsbench [--no-kaggle]
python $P mledojo-rules    --dojo-repo DOJO --data-dir RAW            # rules links still to accept
python $P mledojo-import   --dojo-repo DOJO --data-dir RAW --root ROOT
python $P split            --root ROOT                                # -> ROOT/index.tsv
python $P harbor           --root ROOT --out TASKS [--max-public-mb 2000] [--cpus 4 --memory-mb 16384 --agent-timeout 7200]
python $P export           --tasks-dir TASKS --out train.jsonl
```

All steps can be rerun safely: each one rebuilds its own outputs from its inputs.

## Reference build (CMU Babel, 2026-11-02)

| Stage | MLE-Dojo Kaggle | MLE-Dojo DSBench | Total |
|---|---|---|---|
| downloaded | 81 | 75 | 156 |
| MLE-bench competitions excluded | 26 | 10 | 36 |
| imported | 55 | 65 | 120 |
| usable after `split` checks | 44 | 59 | 103 |
| Harbor tasks built (default `--max-public-mb 2000`) | 42 | 58 | **100** |

101 of the 103 usable tasks are tabular/text and 2 are image tasks. `index.tsv` lists every rejected task and why:
6 too small for a public split, 4 broken metrics, 3 answer leaks, 3 leaderboards not comparable, 1 out of memory.
The three tasks over 2 GB are avazu-ctr-prediction, grupo-bimbo-inventory-demand and
tabular-playground-series-oct-2021. Your numbers depend on which Kaggle rules you accept and on `--max-gb`.

Jeff's copy on Babel is at `/data/user_data/jeffc2/datasets/mle-train/`, with `root/`, `harbor/` and `train.jsonl`.

All data stays outside the repo. Kaggle data is under each competition's rules: use it inside the project only,
and don't commit or publish it. Anyone else rebuilds it with their own Kaggle account (Quick start above).

## Reproducibility

Two people who run `all` with the same settings, and accept the same Kaggle rules, get the same tasks: identical
splits, sample submissions, scores and checks. This was verified by rebuilding from scratch in a fresh environment
and comparing against the reference build. The pieces that make this hold:
- MLE-Dojo is cloned at a pinned commit (`MLEDOJO_COMMIT` in `sources/mledojo.py`), with a warning if a checkout
  differs.
- Grading packages are pinned in `requirements.txt` and in the task Dockerfile.
- The public/private split depends only on the sorted test ids and seed 0.
- MLE-Dojo's own preparation has unseeded randomness. The pipeline seeds `random` and `numpy` from each
  competition's (or DSBench folder's) name before preparing it:
  - DSBench sample submissions are a random row of the answers, picked in filesystem order (now in sorted order);
  - many Kaggle prepare scripts draw random sample submissions (e.g. home-data-for-ml-course,
    mercedes-benz-greener-manufacturing, nyc-taxi-trip-duration);
  - the-nature-conservancy-fisheries-monitoring shuffles its test split.

  Fisheries shuffles a file list in filesystem order, so its test set can still differ between machines. Data
  prepared by an older, unseeded version of the pipeline must be deleted and downloaded again to match.
- Kaggle prepare scripts run with pandas' Python-backed strings. With `pyarrow` installed, pandas 3 would use Arrow
  strings, which sklearn's `train_test_split` can't index (it broke nbme-score-clinical-patient-notes).
- MLE-Dojo's DSBench step extracts with an interactive `unzip` that silently drops files when no terminal is
  attached. The pipeline extracts with Python instead (`sources/dsbench_prepare.py`).

## No MLE-bench competitions

`mlebench_exclude.txt` lists all 82 competition ids in mlebench's registry: the 75 benchmark competitions plus dev
ones. Ids match case-insensitively. A matching task is:
- never imported;
- never split;
- never built;
- never exported (`export` raises an error if one is present).

MLE-Dojo overlaps MLE-bench heavily: 26 of its 81 downloaded Kaggle competitions and 10 of its 75 DSBench tasks are
MLE-bench competitions. If MLE-bench adds competitions, regenerate the file with the command in its header.

## Task format (MLE-bench layout)

```
ROOT/<task_id>/
  config.yaml                           id, source, provenance, metric direction, id column, modality,
                                        split sizes, sanity scores, usable / unusable_reason
  grader/grade.py                       grade(submission: DataFrame, answers: DataFrame) -> float; HIGHER_IS_BETTER
  grader/metric.py, grader/mledojo/     the competition's MLE-Dojo metric class and its base class
  leaderboard.csv                       Kaggle private leaderboard (rank, score)
  prepared/public/                      everything the agent sees: description, train/test inputs, sample submission
  prepared/private/answers.csv          answers for every test row
  prepared/private/public_answers.csv   answers for the public split  (written by `split`)
  prepared/private/private_answers.csv  answers for the private split (written by `split`)
```

`grade.py` uses MLE-bench's grader signature and raises on an invalid submission.

`grader/grade_cli.py` is shared by every task and wraps `grade.py`:
- keeps only the submission rows whose id is in the split;
- treats a missing id, an unreadable file, an exception, or a NaN/inf score as invalid;
- computes the reward;
- prints one JSON line.

## Split and quality checks (`split`)

The test ids are sorted, then shuffled with seed 0, and 30% become the public split. The split doesn't depend on
file order or library version, and ids with several rows stay together.

Each task runs in its own process, so a crash or out-of-memory kill fails only that task.

A task is **usable** only if every check below passes. `index.tsv` gives the reason when one fails.

| Check | What it catches |
|---|---|
| the sample submission and the true answers both grade validly on both splits | broken metrics, format mismatches (e.g. a metric removed in numpy 2) |
| the true answers beat the sample submission on both splits | metric or direction bugs |
| the public split has at least 100 ids | tasks too small for a public leaderboard |
| no public CSV reproduces the answers (constant columns ignored) | answer leaks: we found 3 (ashrae, bioresponse, battlefin) |
| the leaderboard's ranking agrees with the metric direction | inverted metrics |
| the true answers rank at least 0.9 on the Kaggle leaderboard | our metric or test set not comparable to Kaggle's |
| the sample submission ranks at most 0.5 on it | reward already saturated at the start (shopee, covid19-week-1, yelp) |

## Harbor tasks (`harbor`) and training (`export`)

```
TASKS/<task_id>/
  instruction.md          full description, metric direction, compute budget, public/private workflow
  task.toml               metadata: source, direction, split sizes, public_sample (sample submission's public
                          score, the baseline for the first episode); 4 CPUs, 16 GB, 2 h by default
  environment/            Dockerfile (grading packages pinned as in requirements.txt; agent user, sudo rule),
                          bin/validate + bin/validate-root + validate.py (signed scores), score_key (kept across
                          rebuilds), grader + public_answers.csv (root-only in the box), public data (hardlinked)
  tests/                  test.sh (private grading -> reward.txt, final_score.json), grader, private answers,
                          leaderboard
  solution/solve.sh       oracle: submits the sample submission
```

Harbor uploads `tests/` only after the agent finishes, so the private answers and the leaderboard never reach the
agent's box.

The reward is the fraction of the Kaggle private leaderboard the submission beats or ties, so it respects the
metric direction. A missing or invalid submission gets 0, with the reason in `final_score.json`.

`export` writes the files `src/run_training.py` (Miles) reads:
- `HARBOR_TASKS_DIR` = the tasks folder;
- `--prompt-data` = `train.jsonl`, one line per task: `{"prompt": <instruction.md>, "metadata": {"instance_id": <task>, "source": ...}}`.

## Known limitations and open decisions

- **Internet in the box.** Tasks set `network_mode = "public"` so the agent can `pip install`. Many MLE-Dojo test
  sets, like MLE-bench's, come from Kaggle's original *training* data, so an agent with internet could download it
  and find the test answers. To close this, preinstall the packages and turn networking off, or allow only a
  package mirror.
- **Public answers in the box, root-only.** `validate` runs inside the box, so the public answers are there, readable
  only by root; the agent reaches them only through `sudo validate-root`, which refuses protected paths, links and
  files the agent didn't create, and runs the grader with `python3 -I`. Scores are HMAC-signed, so printing
  `score=...` by hand earns nothing. This relies on the sandbox's user separation; the private answers are never in
  the box. Same design as the MLE-bench template in `src/benchmarks/mlebench/`.
- **The reward is an approximation of Kaggle rank.** Where MLE-Dojo re-split Kaggle's training data, our test set
  isn't Kaggle's, so leaderboard position is approximate; MLE-bench has the same limitation. The two
  leaderboard-position checks reject the cases that are clearly not comparable.
- **Sibling competitions.** Only exact MLE-bench ids are excluded. MLE-Dojo also has about 45 Kaggle Playground and
  Tabular Playground episodes. These are different datasets from the three MLE-bench uses (s3e18, Dec 2021,
  May 2022), but some share an original source dataset; for example, forest-cover-type is the original data behind
  Tabular Playground Dec 2021.
- **Compute.** The box has 4 CPUs, 16 GB, no GPU and 2 h. `index.tsv` records `modality` and `public_mb`; image and
  very large tasks may not be solvable in that budget. Filter them with `harbor --max-public-mb` or `--tasks`.
- **Not in the reference build.** benchmark-bond-trade-price-challenge still needs its Kaggle rules accepted.
  global-wheat-detection downloads, but MLE-Dojo's own prepare script fails on it under numpy 2 ("only integer
  scalar arrays can be converted to a scalar index"); it is an image-detection task, so it was left out rather than
  patched.

## Other sources

- **SandMLE** (postponed, `sources/sandmle.py`): synthetic tasks, each shipped with its data generator. As
  distributed, a task has about 30 test rows, too few to split. `sandmle-build` reruns each generator at 35× the
  size. It worked on one task, but its labels are noisy, the medal thresholds are hand-written, and several
  generators are slow. It isn't part of the default build.
- **MLExpert Kaggle-MLE Factory** (not integrated): needs an LLM to write each task's `prepare.py` and `metric.py`,
  plus human review.
