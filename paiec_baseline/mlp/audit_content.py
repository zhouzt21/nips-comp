"""Input-only structural audit; no response labels are loaded."""
import json
from pathlib import Path
import re
import pandas as pd
from .model import _TOKEN_RE, _string


def main():
    split = json.loads(Path("runs/mlp_split_2026.json").read_text())
    records = []
    for directory in split["train_benchmarks"]:
        items = pd.read_parquet(Path("D:/paiec_payload/data") / directory / "items.parquet", columns=["content"])
        if len(items) > 64:
            items = items.sample(64, random_state=2026)
        record = {"benchmark": directory, "sampled_items": len(items), "json": 0,
                  "tokens_above640": 0, "chars_above20000": 0, "opaque_resource_reference": 0}
        lengths, token_counts = [], []
        for value in items["content"]:
            text = _string(value)
            lengths.append(len(text))
            tokens = _TOKEN_RE.findall(text if len(text) <= 20000 else text[:16000]+text[-4000:])
            token_counts.append(len(tokens))
            record["tokens_above640"] += len(tokens) > 640
            record["chars_above20000"] += len(text) > 20000
            record["opaque_resource_reference"] += bool(re.search(r"[a-f0-9]{24,}|\.(?:png|jpe?g|mp4|wav)\b", text))
            try:
                obj = json.loads(text)
                record["json"] += isinstance(obj, (dict, list))
            except (ValueError, TypeError):
                pass
        record["mean_chars"] = sum(lengths)/max(1,len(lengths))
        record["mean_tokens_before_cap"] = sum(token_counts)/max(1,len(token_counts))
        records.append(record)
    path = Path("runs/research_loop/content_audit.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed":2026,"labels_read":False,"max_items_per_benchmark":64,
                                "benchmarks":records}, indent=2), encoding="utf-8")
    print(json.dumps({key:sum(r[key] for r in records) for key in
                      ["sampled_items","json","tokens_above640","chars_above20000","opaque_resource_reference"]}))


if __name__ == "__main__":
    main()
