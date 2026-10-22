"""Package the smoothed-mean control as a standalone submission ZIP."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def build_zip(output: Path | None = None) -> Path:
    root = Path(__file__).resolve().parents[1]
    output = output or root / "dist/smoothed_mean_strength2.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {name: root / "smoothed_mean" / name for name in ("model.py", "README.md")}
    files["LICENSE"] = root / "LICENSE"
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, source in files.items():
            entry = ZipInfo(name)
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, source.read_bytes())
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    path = build_zip(args.out)
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)")
