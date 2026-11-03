# Task

{description}

# Metric direction

The score is **{direction}**.

# Workflow

Work iteratively: build a model, check it on the public leaderboard with `validate <file.csv>`, improve, resubmit.

1. The data is in `/workspace/public/` (training data, the test inputs, and `{sample_submission}`, the exact
   submission format). Use only files in `/workspace/public/`.
2. Predict **every** row of the test set, in the format of `{sample_submission}` (same columns, header, ids).
3. `validate <your_file.csv>` scores your file on the **public leaderboard**, a fixed hidden subset of about 30% of
   the test rows, and prints `score=...`. Call it after every material improvement; the run tracks your
   per-submission scores and rewards improving on your previous best public score.
4. Your **final** submission must be written to `/workspace/predictions.csv`. It is graded at the end on the
   **private leaderboard**, the other ~70% of the test rows, which you never see a score for. Avoid overfitting to the
   public leaderboard.

# Rules
- Only `validate` may be used to get scores. Do not look for, read, or reconstruct the answers.
- Compute: {cpus} CPUs, {memory_gb} GB RAM, no GPU, {hours} hours in total. You may install packages with pip.
- Keep iterating until your score stops improving, then make sure `/workspace/predictions.csv` holds your best model's
  predictions for the full test set.
