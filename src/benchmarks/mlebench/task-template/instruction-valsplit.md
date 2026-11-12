{description}

# Workflow

Work iteratively: build a model, check it on the validation set with `validate <file.csv>`, improve, and resubmit. Track your score over submissions.

1. The competition data is already prepared at `/workspace/public/`: the labeled training data, the unlabeled test
   inputs you must predict, and `sample_submission.csv` (the exact submission format, one row per test id).
   Use only files in `/workspace/public/`.
2. Train an ML model on the training data to minimize the competition metric.
3. Predict **every** test id and write the predictions in the format of `sample_submission.csv`
   (same columns/order/headers). You may name intermediate submission files however you like.
4. **For every material improvement, score your file on the validation set**:
   `validate <your_submission_file.csv>` (e.g. `validate preds_v2.csv`). A fixed hidden subset of the test ids is
   the validation set; `validate` grades your predictions for those ids and prints `score=...`. **Do this after every
   material improvement**, not just at the end: the run tracks your per-submission scores and rewards a score that
   improves on your previous best.
5. When satisfied, write your **final** submission to `/workspace/predictions.csv` (this exact path is graded at the
   end, on the other test ids). Match the `sample_submission.csv` format exactly.

# Rules
- **Never read or use the hidden answers**: they are held out and strictly off-limits. Only `validate <file.csv>`
  grades against them, and it returns only a score. Reading the answers is cheating.
- Only the submitted `predictions.csv` is graded; each validate call is logged.
- Install any packages you need with `pip install --user <package>` (you run as the unprivileged user `agent`; CPU-only environment, no GPU in the default setup).
- `validate <file.csv>` (or `python validate.py <file.csv>`) is the only way to score a submission; the grader, the answers and the leaderboard are not accessible to you.
- Improve on your prior best validation score across the run.
