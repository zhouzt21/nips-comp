# Packaging compatibility changes, before final ZIP acceptance

The first dynamic official ZIP checker run reached inference but failed to unlink memory-mapped safetensors on Windows. The packaged constructor now clones tensor storage once after load; parameter values and encode_items remain unchanged. This releases file mappings for official temporary-directory cleanup.

The first raw-input acceptance attempt was stopped after five Benchmarks to avoid redundant re-encoding. The packaging wrapper now uses a bounded 2,048-entry LRU of input-only feature vectors (item_content, item_features, interactors). No Subject, label, budget, response or prediction is stored in this cache. All label contexts remain local to each call. There is no disk cache. This is a runtime optimization; it changes neither the fixed model nor the evaluation protocol. Earlier request-local-only cache wording in CONTRACT.md records the initial implementation; this addendum governs the final wrapper. Final ZIP acceptance is restarted in fresh processes after rebuilding.

The first interrupted acceptance logs are retained under logs/attempt1. All final readiness and resource claims use the rebuilt ZIPs and final completed runs.
