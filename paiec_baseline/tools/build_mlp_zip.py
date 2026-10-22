"""Package the fitted, offline MLP as a local Codabench submission ZIP."""

import argparse
import ast
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def build_zip(output: Path | None = None, *, weights: Path | None = None,
              readme: Path | None = None) -> Path:
    root = Path(__file__).resolve().parents[1]
    output = output or root / "dist/mlp_candidate.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {name: root / "mlp" / name for name in ("model.py", "weights.npz", "README.md")}
    if weights is not None:
        files["weights.npz"] = weights
    metadata = files["weights.npz"].with_suffix(".json")
    if metadata.is_file():
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if record.get("feature_mode", "all") != "all" or record.get("network_kind", "mlp") != "mlp":
            raise ValueError("This packager supports legacy all-feature MLP weights only; research variants require a matching portable predictor")
    if readme is not None:
        files["README.md"] = readme
    files["LICENSE"] = root / "LICENSE"
    for source in files.values():
        if not source.is_file():
            raise FileNotFoundError(f"Cannot package MLP without {source}")
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, source in files.items():
            entry = ZipInfo(name)
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            content = source.read_bytes()
            if name == "model.py":
                # Research dispatch is not called by the legacy predict().
                # Remove its optional relative imports from the standalone ZIP.
                text = content.decode("utf-8")
                nodes = [node for node in ast.parse(text).body
                         if isinstance(node, ast.FunctionDef) and node.name == "encode_mode"]
                lines = text.splitlines(keepends=True)
                for node in reversed(nodes):
                    del lines[node.lineno - 1:node.end_lineno]
                content = "".join(lines).encode("utf-8")
            archive.writestr(entry, content)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--weights", type=Path, help="override the fitted NPZ artifact")
    parser.add_argument("--readme", type=Path, help="override the archive README")
    args = parser.parse_args()
    path = build_zip(args.out, weights=args.weights, readme=args.readme)
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)")
