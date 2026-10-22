# R18b / V2-09: restore omitted text while holding the encoder fixed

Registered2026-10-22 after input-only R18a, before new embeddings and scores. Among3600deterministically sampled items,1783exceed126wordpieces,1220exceed254; canonical text has3243distinct strings but retained128windows only3108. These count equivalences, not proven task-relevant distinctions.679strings match image-reference syntax; no claim of pixel access follows.

The material text omission justifies one controlled representation experiment. Same pinned MiniLM, parser/canonicalization, mean pooling, normalization,384dimensions and localridge. Compare existing128window with256window:254contenttokens(first192+last62), plusCLS/SEP. Lexical control receives the exact same decoded retained window. No image download, new pretrained model, extra training corpus or supervised encoder update.

Use R7's source episodes(seed73,max25pairs/BM), ridge gridoff/16/4/1/.25 selected independently for semantic256 andtext256, allthree selections before outer scoring. Compare old128methods with their already frozen selections; same outerseed0,targetpairs,evaluationresponses. If gains are weak/negative, do not proceed to larger window/model simply by default. If256helps bothlexical/semantic, input availability is a likely component; semantic-only gain suggests representation-by-context interaction, not a pure length effect. Formal mechanism attribution still limited by token context effects.

Before scoring, independent new-window encoder with128setting must match original128vectors/hash features on a small batch within2e-6. Actual model inputs at most256; tokenizing longer strings for truncation is allowed and not a forward pass. R18a PowerShell wrapper reported a stderr warning despite completeJSON; a subprocess return-code rerun will record actual completion separately.

Record benchmark/seed/selection choices, runtime, cache fingerprint and modelrevision. Preserve oldcode/caches. Frozen0.005andallothergates unchanged; scores from this run remain reused-public-fold evidence.
