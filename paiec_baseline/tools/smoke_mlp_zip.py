"""Exercise the packaged MLP from a clean, isolated Python process."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile


PROBE = r'''
import importlib.util
import json
import math
from pathlib import Path
import sys
from time import perf_counter

root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("candidate_model", root / "model.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
subject = {"normalized_name": "unseen-model", "provider": "unseen-provider"}
item = {"benchmark_id": "anonymous-unseen", "item_content": "Answer a new question.",
        "item_features": "", "interactors": ""}
target = [subject, item]
other = [[subject, dict(item, benchmark_id="another-anonymous-benchmark")], 0]
labels = [[[subject, dict(item, item_content=f"Question {i}")], int(i % 2 == 0)]
          for i in range(31)]
start = perf_counter()
p0 = model.predict(target, [])
assert p0 == model.predict(target, [other])
scores = [model.predict(target, labels[:budget]) for budget in (0, 1, 3, 7, 15, 31)]
assert scores[0] == p0
assert all(type(value) is float and math.isfinite(value) and 0 <= value <= 1
           for value in scores)
assert model.predict([{}, {"item_content": "missing attributes"}], []) >= 0
assert model.predict(target, labels[:3]) == scores[2]
print(json.dumps({"budget_probabilities": scores,
                  "elapsed_seconds": perf_counter() - start}))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_zip", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="paiec-mlp-smoke-") as temp:
        with ZipFile(args.submission_zip) as archive:
            archive.extractall(temp)
        result = subprocess.run(
            [sys.executable, "-I", "-c", PROBE, temp], cwd=temp,
            text=True, capture_output=True, check=True,
        )
        print(result.stdout.strip())


if __name__ == "__main__":
    main()
