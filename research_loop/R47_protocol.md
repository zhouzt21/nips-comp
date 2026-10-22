# R47: public multimedia observability audit

Read-only contract audit after R46; this document records the audit, not a preregistered performance experiment. No model scores, pixel downloads or new pretrained model were selected.

Question: are image bytes present in the permitted public data, and are equivalent bytes explicitly accessible through the hosted prediction interface? These are separate questions. Inspect the current public dataset revision, asset table listings, curation schema, official predictor specification and public starting-kit entry. Do not access private reference/test data or infer their contents.

Sources: https://aims.stanford.edu/competition ; https://huggingface.co/datasets/aims-foundations/measurement-db ; public curation guidelines exported in literature/raw/v3_curation_guidelines.txt. The public starting-kit link resolves to competition17828. Its page and public metadata API fail TLS in requests; the web tool also cannot access the API. Do not disable TLS validation to bypass this.

Boundary: relative paths in item_content and available training assets do not establish a runtime asset resolver. Missing documentation does not prove the hosted system cannot provide pixels. Keep this branch blocked on an explicit public runtime contract; do not make a deployable pixel-method claim.
