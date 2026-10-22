# R18 / V2-09: audit loss at the input interface before changing representation

Registered2026-10-22 before input audit. R17 bounds repair of current predictions; it does not bound information available in the input. R7 deliberately limited inputs to126wordpieces+special tokens, while its model card permits256wordpieces. Many tasks are multimodal. These are hypotheses about omitted information, not automatic reasons to buy a bigger encoder.

R18a: deterministic sample of up100 items per original outer benchmark directory, seed123. Read item contents and public metadata, no response labels. Record raw/canonical/retained-window equivalence counts, canonical token lengths, fractions beyond126/254, and metadata modality. Text patterns for image references are only syntactic flags, not proof that pixels are accessible in the hosted interface. Separate genuinely repeated input from new equivalence introduced by parsing/windowing.

If a material number of items lose text between126and254, a matched128/256window experiment becomes justified. Keep the same pretrained model, parser, parameters and semantic/lexical controls; no model-family switch. If text loss is scarce, do not enlarge the window; investigate other available fields or modality access instead. Any performance experiment needs a separate R18b preregistration before fitting/scoring.

Do not modify R7 semantic_features.py or invalidate historical cache keys. A new encoder variant must preserve old artifacts and permit exact128replay.
