{description}

# Workflow

Work iteratively: build a model, score your submissions on the held-out test split with `validate <file.csv>`, improve, and resubmit. Track your score over submissions.

1. The competition data is already prepared at `/workspace/public/`:
   - `train.csv` — the labeled training data,
   - `test.csv` — the unlabeled rows you must predict,
   - `sample_submission.csv` — the exact submission format.
   Use only files in `/workspace/public/`.
2. Train an ML model on the prepared `train` split to minimize the competition metric.
3. Write predictions to a submission file in the format of the competition
   `sample_submission.csv` (same columns/order/headers).
4. **For every material improvement, get your score on the held-out split**:
   `validate <your_submission_file.csv>`
   (e.g. `validate preds_v2.csv`) — it grades that file against the held-out
   answers and prints `score=...`. You may name intermediate submission files
   however you like. **Do this after every material improvement**, not just at
   the end — the run tracks your per-submission scores and rewards a score that
   improves on your previous best.
5. When satisfied, write your **final** submission to `/workspace/predictions.csv`
   (this exact path is what gets graded at the end). Match the `sample_submission.csv`
   format exactly.

# Rules
- **Never read or use the `private` test split / answers** — they are held-out and
  strictly off-limits. Only the `validate <file.csv>` command grades against them,
  and it returns only a score. Reading the answers is cheating.
- Only the submitted `predictions.csv` is graded; each validate call is logged.
- Install any packages you need (CPU-only environment; no GPU in the default setup).
- Improve on your prior best validation score across the run.
- Keep iterating until you have a strong score; write your final `predictions.csv` once satisfied.
