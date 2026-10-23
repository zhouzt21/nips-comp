# ADeLe source evidence, checked 2026-10-23

- Published journal article: Zhou et al., Nature 652, 58–67 (2026), online 2026-04-01; DOI 10.1038/s41586-026-10303-2. Current AnySearch `academic.search` returned the exact DOI and Cambridge repository. Search is discovery only.
- Full official paper locally read: `D:/nips-comp/paiec-dataset/reproduction_sources/adele/nature_paper.pdf`, SHA256 `74590d2e0c3912be5e40af0d7e268ac2431f0f73f19dcaa09fa3f00ed80b2ebb`.
- Official supplement: same directory `nature_supplementary.pdf`, SHA256 `94067d561a9ef2c44e51c8f851a07ce0ef16192484345e4bceb519a555e1bdee`.
- Exact locations: main PDF p13 annotation and inter-rater analysis; pp14–15 assessors/metrics; supplement p58 Table23 prompt. Local pdftotext lines 1019–1059 and 1108–1165 were re-read this run.
- Author repository: https://github.com/Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation commit `816321f327dfe66910b72b514156aaa296b15f00`; HEAD verified locally. Native code search did not retrieve the target; original fixed clone is evidence.
- Code: `src/adele/annotation/prompts.py:16` builds exact prompt; `rubrics/catalog.py` and `rubrics/data_v1/AS.txt,MCr.txt,QLl.txt,KNf.txt` supply unmodified rubric. `src/adele/analysis/prediction.py:193` exposes RF 100 estimators, minimum split50, entropy. R73 independently reconstructed all512 old prompts exactly.
- Supported transfer: demand annotation is derived from task/rubric rather than success outcome; source-grouped assessment is needed for OOD. Not supported: Luna grades equal original GPT4o grades, four dimensions equal original18+UG, ridge equals author RF, PAIEC Brier gain guaranteed.
- All adaptations and unavailable native runtime parameters are disclosed in `reports/demand_annotation_findings.md`. No external ADeLe benchmark labels are used for PAIEC training.
