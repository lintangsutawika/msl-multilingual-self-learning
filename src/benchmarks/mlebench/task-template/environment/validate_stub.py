"""Grade a submission:  validate <file.csv>   (same as: python validate.py <file.csv>)

The grader and the held-out answers are root-only; this just runs `validate`.
"""
import os
import sys

os.execvp("validate", ["validate", *sys.argv[1:]])
