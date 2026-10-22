# Runtime note

The completed original run emitted repeated pandas lexsort PerformanceWarning messages while looking up historical scores. These were performance warnings; all pair/seed/budget assertions passed, exit_code=0, and the run finished in150.82seconds. Exact executed source is preserved in code/ and provenance.json. After completion the working helper adds sort_index() to avoid log spam in future reruns. This does not change score values or model parameters. The original log is retained, not deleted or replaced by a clean fictional run.
