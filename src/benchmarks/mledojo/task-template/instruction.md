# Task

{description}

# Metric direction

The score is **{direction}**.

# Workflow

Work iteratively: build a model, check it on the validation set with `validate <file.csv>`, improve, resubmit.

1. The data is in `/workspace/public/` (the labeled training data, the unlabeled test inputs you must predict, and
   `{sample_submission}`, the exact submission format, one row per test id). Use only files in `/workspace/public/`.
2. Predict **every** row of the test set, in the format of `{sample_submission}` (same columns, header, ids).
3. `validate <your_file.csv>` scores your file on the **validation set**, a fixed hidden subset of the test ids, and
   prints `score=...`. Call it after every material improvement; the run tracks your per-submission scores and
   rewards improving on your previous best validation score. (`python validate.py <file.csv>` in `/workspace` does
   the same.)
4. Your **final** submission must be written to `/workspace/predictions.csv`. It is graded at the end on the other
   test ids, which you never see a score for. Avoid overfitting to the validation set.

# Rules
- Only `validate` may be used to get scores. Do not look for, read, or reconstruct the answers.
- The answers and the grader are not accessible to you; `validate` returns only a score.
- You run as the unprivileged user `agent`. Install packages with `pip install --user <package>`.
- Compute: {cpus} CPUs, {memory_gb} GB RAM, no GPU, {hours} hours in total.
- Keep iterating until your score stops improving, then make sure `/workspace/predictions.csv` holds your best model's
  predictions for the full test set.
