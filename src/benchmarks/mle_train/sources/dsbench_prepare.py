"""Run MLE-Dojo's prepare/dsbench.py with a robust download + extraction step.

MLE-Dojo's own step shells out to `unzip` without -o. The archive has duplicate __MACOSX entries, so unzip stops to
ask whether to overwrite; without a terminal it answers "None" and leaves the extraction incomplete, and the later
steps crash. Here the zip is streamed to disk and extracted with zipfile (skipping __MACOSX, never prompting), then
MLE-Dojo's main() runs unchanged for every later step.

MLE-Dojo also fills each sample submission with one random row of the answers, using unseeded numpy and filesystem
(os.listdir) order, so baselines differed between builds. Here each folder gets its own seed derived from its
name and folders are processed in sorted order, so every machine builds identical sample submissions.

    python dsbench_prepare.py <mle-dojo repo> <raw dir> <prepared dir>
"""

import shutil
import sys
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np

URL = "https://huggingface.co/datasets/liqiang888/DSBench/resolve/main/data_modeling/data.zip"


def fetch_and_extract(base_dir: str) -> None:
    base = Path(base_dir)
    zip_path = base / "data.zip"
    part = zip_path.with_suffix(".zip.part")
    if not zip_path.exists():
        print(f"Downloading {URL} ...", flush=True)
        with urllib.request.urlopen(URL, timeout=120) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, length=16 << 20)
        part.rename(zip_path)
    print("Extracting ...", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            p = Path(m.filename)
            if p.parts and p.parts[0] == "__MACOSX" or p.is_absolute() or ".." in p.parts:
                continue
            z.extract(m, base)
    zip_path.unlink()
    print("Data extraction complete.", flush=True)


def main() -> None:
    dojo, raw, prepared = (Path(a).resolve() for a in sys.argv[1:4])
    sys.path[:0] = [str(dojo), str(dojo / "prepare")]
    import dsbench  # MLE-Dojo's prepare/dsbench.py

    dsbench.download_and_extract_data = fetch_and_extract
    modify = dsbench.modify_submission_files

    def seeded_modify(folders, data_resplit_dir):
        for f in sorted(folders):
            np.random.seed(zlib.crc32(f.encode()))
            modify([f], data_resplit_dir)

    dsbench.modify_submission_files = seeded_modify
    sys.argv = ["dsbench.py", "--raw-dir", str(raw), "--prepared-dir", str(prepared)]
    dsbench.main()


if __name__ == "__main__":
    main()
