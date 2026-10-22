"""Offline smoothed empirical mean control for PAIEC."""

STRENGTH = 2.0


def predict(input: list[dict], labeled: list | None = None) -> float:
    """Use a 0.5 prior with two pseudo-observations for the current group."""
    subject, item = input
    if not labeled:
        return 0.5
    benchmark = item.get("benchmark_id")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("A nonempty benchmark_id is required with acquired labels")
    successes = 0
    count = 0
    for (observed_subject, observed_item), label in labeled:
        if observed_subject != subject or observed_item.get("benchmark_id") != benchmark:
            continue
        if label not in (0, 1):
            raise ValueError("Acquired response must be binary")
        successes += int(label)
        count += 1
    return float((0.5 * STRENGTH + successes) / (STRENGTH + count))
