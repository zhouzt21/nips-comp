"""Input-only JSON text extraction, without benchmark names or outcome data."""
from __future__ import annotations
import json
import re

_WORDS = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SKIP = {"id", "item_id", "task_id", "database_id", "location", "url", "path", "source_image_path"}
_PRIMARY = {"question", "prompt", "text", "instruction", "input", "problem", "query"}


def extract_text(value: object) -> str:
    raw = "" if value is None else str(value)
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return raw
    if not isinstance(obj, (dict, list)):
        return raw
    leaves = []

    def visit(node, key="content", priority=1):
        if isinstance(node, dict):
            role = str(node.get("role", "")).lower()
            inherited = 0 if role == "system" else 3 if role == "user" else priority
            for name, child in node.items():
                name = str(name).lower()
                if name in _SKIP or name.endswith("_id"):
                    continue
                rank = 0 if "system" in name or role == "system" else (3 if name in _PRIMARY else inherited)
                visit(child, name, rank)
        elif isinstance(node, list):
            for child in node[:256]:
                visit(child, key, priority)
        elif isinstance(node, (str, int, float, bool)):
            text = str(node)
            # A long hexadecimal resource digest is not the resource contents.
            text = re.sub(r"\b(?=[a-fA-F0-9]{24,}\b)(?=[a-fA-F0-9]*[a-fA-F])(?=[a-fA-F0-9]*[0-9])[a-fA-F0-9]+\b", "<resource>", text)
            tokens = _WORDS.findall(text)
            if tokens:
                leaves.append((priority, len(leaves), key, tokens))

    visit(obj)
    leaves.sort(key=lambda row: (-row[0], row[1]))
    output = []
    for priority, _, key, tokens in leaves:
        cap = {0:64, 1:128, 3:384}.get(priority, 128)
        selected = tokens if len(tokens) <= cap else tokens[:3*cap//4] + tokens[-cap//4:]
        output.extend([key, *selected])
    # The normal encoder performs the same final 640-token bound. Empty JSON
    # gets an explicit marker instead of recovering dropped opaque IDs.
    return " ".join(output) if output else "<empty_structure>"
