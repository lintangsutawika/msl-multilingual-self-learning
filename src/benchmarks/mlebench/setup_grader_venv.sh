#!/usr/bin/env bash
# Build the "grader venv": Python 3.12 + MLE-bench at the pinned revision + pandas<3.
#
#   src/benchmarks/mlebench/setup_grader_venv.sh /path/to/mlebench-grader-venv
#
# Why a second venv: the repo's .venv has pandas 3, and several of MLE-bench's prepare.py scripts break under it
# (`.unique()` returns an ArrowStringArray that train_test_split rejects). Preparing data, making the validation split,
# the leak audit and the host-side `validate` grader all run here, with the same MLE-bench revision and pandas<3 as the
# task image's grader, so scores match the image's. Set GRADER_PY=<venv>/bin/python in site.env.
set -euo pipefail
V=${1:?usage: $0 VENV_DIR}
REV=$(sed -n 's/^MLEBENCH_REV = "\(.*\)"/\1/p' "$(dirname "$0")/adapter.py")  # same pin as the task image
uv venv -q -p 3.12 "$V"
# --no-deps: MLE-bench's own dependencies pull TensorFlow, which no grader uses. The list below is what its prepare
# and grading modules import (as in task-template/environment/Dockerfile), plus kaggle for downloading.
uv pip install -q -p "$V/bin/python" --no-deps "git+https://github.com/openai/mle-bench.git@$REV"
uv pip install -q -p "$V/bin/python" "pandas>=2.2,<3" numpy scipy scikit-learn tqdm diskcache tenacity pyyaml \
    appdirs py7zr levenshtein pycocotools pyquaternion shapely pillow pymongo requests "kaggle==1.6.17"
"$V/bin/python" -c "import importlib.metadata as m, mlebench.registry, pandas; m.version('kaggle'); print('grader venv ok: pandas', pandas.__version__)"
