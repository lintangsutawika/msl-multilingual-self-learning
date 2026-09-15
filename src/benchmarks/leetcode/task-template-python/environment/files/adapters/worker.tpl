import contextlib
import json
import sys
from typing import *
from collections import *
from functools import *
from itertools import *
from heapq import *
from bisect import *
from math import *
import collections, functools, itertools, heapq, bisect, math, random, string
with contextlib.redirect_stdout(sys.stderr):
    exec(open("solution.py").read(), globals())
    candidate = TARGET
for line in sys.stdin:
    args = json.loads(line)
    with contextlib.redirect_stdout(sys.stderr):
        result = candidate(*args)
    print(json.dumps(result, allow_nan=False), flush=True)
