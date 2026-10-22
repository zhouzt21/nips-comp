"""Sequential CPU experiment runner with durable commands, hashes, and timings."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(command, log, records):
    started = time.time()
    print(f"START {log}", flush=True)
    env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"}
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, env=env)
    record = {"command": command, "log": str(log), "started_unix": started,
              "elapsed_seconds": time.time()-started, "exit_code": result.returncode}
    with records.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record) + "\n")
    if result.returncode:
        raise RuntimeError(f"Failed: {log}; inspect saved log")
    print(f"DONE {log}: {record['elapsed_seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", choices=["R1", "R2"], default="R1")
    args = parser.parse_args()
    root = Path("runs/research_loop") / args.round
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "code"
    snapshot.mkdir(exist_ok=True)
    hashes = {}
    for file in ["model.py", "train.py", "evaluate.py", "run_loop_round.py"]:
        source = Path("mlp") / file
        data = source.read_bytes()
        dest = snapshot / file
        if dest.exists() and dest.read_bytes() != data:
            raise RuntimeError("Code changed during round; use a new round ID")
        dest.write_bytes(data)
        hashes[str(source)] = hashlib.sha256(data).hexdigest()
    (root / "code_sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    candidates = [("block", "block", "mlp"), ("linear", "all", "linear")] if args.round == "R1" else [("structured", "structured", "mlp")]
    for fold in range(3):
        split = f"runs/mlp_cv/fold{fold}_split.json"
        weights = [f"original=runs/mlp_cv/fold{fold}/weights.npz"]
        for name, mode, network in candidates:
            folder = root / f"{name}_fold{fold}"
            folder.mkdir(exist_ok=True)
            out = folder / "weights.npz"
            if not out.exists() or not out.with_suffix(".json").exists():
                run([sys.executable, "-m", "mlp.train", "--data-root", "D:/paiec_payload/data",
                     "--split-file", split, "--out", str(out), "--seed", "2026",
                     "--max-rows-per-benchmark", "400", "--epochs", "30", "--batch-size", "512",
                     "--learning-rate", "0.001", "--weight-decay", "0.0001",
                     "--feature-mode", mode, "--network-kind", network, "--internal-validation"],
                    folder / "train.log", root / "commands.jsonl")
            weights.append(f"{name}={out}")
        evaluation = root / f"fold{fold}_evaluation.json"
        if not evaluation.exists():
            command = [sys.executable, "-m", "mlp.evaluate", "--data-root", "D:/paiec_payload/data",
                       "--split-file", split, "--role", "development", "--seeds", "1",
                       "--max-evaluation-items", "64", "--max-pairs-per-benchmark", "100",
                       "--alphas", "2", "4", "--out", str(evaluation)]
            for weight in weights:
                command += ["--weights", weight]
            run(command, root / f"fold{fold}_evaluate.log", root / "commands.jsonl")
    print("ROUND_FINISHED", flush=True)


if __name__ == "__main__":
    main()
