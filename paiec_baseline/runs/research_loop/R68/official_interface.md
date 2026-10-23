[Skip to main content](https://aims.stanford.edu/competition#main-content)

## Search AIMS

Find pages, publications, events, course projects, and software. Results update as you type; press Enter to open the first result.

# The Predictive AI Evaluation Competition at NeurIPS 2026

[aimslab@cs.stanford.edu](mailto:aimslab@cs.stanford.edu)[Submit on Codabench](https://aims.stanford.edu/competition/submit)[Training Dataset](https://huggingface.co/datasets/aims-foundations/measurement-db)

Session venue

Sydney, Australia

Session date

Friday, 11 December 2026

Session format

In person · 08:00 to 10:15 Sydney time

## Overview.

Decision makers in enterprises, hospitals, and schools need to assess the suitability of AI systems for their intended applications, often with limited evaluation resources. Reliable predictive evaluators could inform these decisions by estimating system performance on previously unevaluated items from partial observations. This competition provides a shared testbed for studying the predictability of AI responses and the amount of evidence required for accurate prediction.

A subject is an AI system represented by a fixed configuration of model, evaluation harness, harness version, and reasoning effort. The same model evaluated under two harnesses constitutes two distinct subjects. An item is a question or task, together with its features and any tools or agents involved in the interaction. A response records the outcome of one run of a subject on an item, scored 1 if the subject answered correctly and 0 otherwise. A target is a response to be predicted; a subject that attempted an item three times contributes three separate targets.

The task is to predict unobserved responses from the attributes of AI systems and benchmark items. For each subject–item pair, the submitted program receives these attributes and estimates the probability of a response of 1, without running the subject on the item. The evaluation is a benchmark-level cold start: held-out items are drawn from benchmarks assigned to the private test pool. To provide a consistent evaluation process for all teams, the platform selects the subjects for each submission. The planned NeurIPS competition session includes five oral presentations totaling 60 minutes, followed by a 60-minute poster session. All teams are welcome to submit their work for consideration for an oral or poster presentation. The remaining 15 minutes are reserved for opening remarks and transitions.

Public training benchmarksHeld-out benchmarks

Subjects (AI systems)Items (benchmark questions)

- Response = 1
- Response = 0
- Unobserved

Rows represent AI systems; columns represent benchmark items. Blue and purple cells show observed training responses; cyan cells show test outcomes unknown at training time. The submitted program predicts the probability of response 1 from the subject and item attributes and available evidence. The shared subjects shown here are illustrative; subject coverage may differ between pools.

## Participation.

### Register your team

To get started, please complete the [team registration form](https://docs.google.com/forms/d/e/1FAIpQLSevCdcDO1TaWqWUY7vRMljoYE97PJXC-X3Vg3ItD3kRi8XUmQ/viewform?usp=header). Submit one form per team, listing your primary contact as Member 1. You’ll need your team name and each member’s name, affiliation, email address, Codabench username, and Hugging Face username. Registration is free.

### Community & Support

Participants are welcome to join the [Discord community](https://discord.gg/W2rmjJdcm) to ask questions and connect with fellow participants. The organizers also offer office hours by appointment and are happy to help with questions about methods, implementation, or participation. Appointments can be requested by emailing [aimslab@cs.stanford.edu](mailto:aimslab@cs.stanford.edu).

Final codebase and technical report deadline: 30 October 2026 at 23:59 Anywhere on Earth (AoE).

- Eligibility. Each participant may join only one team. Cross-institutional teams are encouraged, and teams may include up to six members. The organizing team and their immediate collaborators may submit entries for comparison.
- Submissions. Teams are welcome to submit multiple entries during development. For final evaluation, teams must submit their codebase and technical report by the submission deadline and provide their selected Codabench submission ID in the OpenReview submission form.
- Training data. All data sources listed in the benchmark inventory are permitted for competition-specific training, development, and curation. Participants must not use data sources outside that inventory for these purposes. See the [benchmark inventory](https://docs.google.com/spreadsheets/d/1T69QyIZJRWX3d-Tmuh_lcFwtpqv9ytcLxfLWKn6gopI/edit?gid=0#gid=0).
- Runtime. Network access during scoring is limited to inference APIs from Anthropic, OpenAI, Google, xAI, DeepSeek, Mistral, Together AI, and Groq. Web search, browsing, and grounding are prohibited, including when offered through an approved provider's API. Participants carry out initial model training before submission. The submitted program may then adapt during evaluation using the supplied attributes and labels revealed within its budget.
- Disclosure. Disclose all training sources in the technical report. The use of LLM-based coding assistants during development is permitted. We kindly ask participants to disclose any such use in the technical report.
- Presentation consideration. Consideration for an oral or poster presentation requires a technical report and release of the code used to develop and run the submission, along with any permitted training data that differ from measurement-db. Technical report quality is an important factor in selection for both oral and poster presentations. The technical report should use the NeurIPS template and contain approximately 4–8 pages of main text, excluding references and appendices. References and appendices have no page limit. Technical reports do not need to be anonymous. The report must clearly describe the data, submitted approach, implementation and experimental setup, results, and limitations, with instructions sufficient to rerun the method. By the submission deadline, teams must submit the technical report, including links to these released materials, through [OpenReview](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FPAIEC). For assessment criteria, see [Summative evaluation](https://aims.stanford.edu/competition#summative-evaluation).
- Conduct. Participants are encouraged to discuss methods and share code and permitted training data across teams, with shared contributions acknowledged in the technical report. Formative feedback may guide development, but participants must not extract or share private test data or attempt to identify the subjects or benchmarks behind anonymous feedback IDs for use in development. The submitted program may use the attributes and acquired labels supplied through the evaluation interface to adapt during evaluation.

## Competition.

### Predictor Interface.

#### Package and submit

Download the [starting kit from Codabench](https://www.codabench.org/competitions/17828/). Place the files below at the root of your ZIP, run the local checks, and upload it to Codabench. Bundle local models or declare them in `models.txt` for prefetching.

```
submission.zip/
  model.py              required: predict(input, labeled)
  labeling.py           optional: test-time adaptation
  models.txt            optional: HuggingFace repos to prefetch
  requirements.txt      optional, organizer policy gated
  any_other_files/      artifacts, fitted state, templates,...
```

```
# Run from the starting kit directory
(cd my_submission && zip -r ../my_submission.zip .)

python tools/check_submission_zip.py my_submission.zip
python tools/run_smoke_test.py my_submission/
```

Troubleshooting+

`10/11`: invalid `model.py`; `12`: invalid `labeling.py`; `40`: prediction exception; `41`: call timeout; `42`: invalid probability; `50`: total timeout. Hosted logs hide raw `stdout`/`stderr`; use the local checks above for debugging.

At each label budget, the platform calls `predict(input, labeled)` for each distinct `[subject, item]` input. Repeated responses with identical inputs reuse that prediction. `input` contains the two dictionaries listed below; `labeled` contains the evidence allowed at that budget. Every field value is a plain string. Missing values are represented by the empty string `""`.

```
# model.py (required)
def predict(input: list[dict], labeled: list) -> float:
    """Predict P(correct) from the supplied input and revealed labels."""
    subject, item = input
    ...
```

`input[0]` describes the subject, the AI system whose response is being predicted. It is a dictionary with exactly the following keys:

- normalized\_name: Canonical name of the AI system.
- provider: Organization that released the model.
- release\_date: When the model was released.
- access\_date: When the model response was recorded.
- harness: Agent scaffold or CLI driving the model.
- harness\_version: Version of that harness.
- reasoning\_effort: Reasoning effort setting for the run.
- subject\_features\_extra: Further settings recorded for the model run.

`input[1]` describes the item, the question the subject attempted. It is a dictionary with exactly the following keys:

- item\_content: The task text presented to the subject.
- item\_features: Additional attributes of the item.
- interactors: Tools or agents involved in the interaction.
- benchmark\_id: A stable, anonymous identifier for the item's benchmark, shared by target items and acquired examples from the same benchmark.

The prediction function must return a finite probability in `[0, 1]`, preferably as a native Python `float`. The runtime applies `float()` to the result and rejects conversion failures, `NaN`, infinities, and out-of-range values. The complete example below predicts `0.5` for every target.

```
# model.py
def predict(input: list[dict], labeled: list) -> float:
    return 0.5
```

### Test-time Adaptation.

Test-time adaptation allows the submitted program to incorporate a limited number of observed test responses, while an acquisition function determines which outcomes to reveal. For each subject–benchmark pair with at least 80 distinct items, items are randomly split 50/50 into acquisition and evaluation pools. The split is fixed across submissions and budgets, and all recorded responses for an item stay together. Acquisition items are never scored; evaluation outcomes are never revealed.

Every submission is evaluated at label budgets of 0, 1, 3, 7, 15, and 31 per pair on the same evaluation items. One acquisition trajectory provides nested label sets, so labels available at smaller budgets remain available at larger ones. Each evaluation receives only its allowed evidence; at budget 0, `labeled` is empty.

#### Acquisition function

Candidates arrive one at a time in a fixed random order. An optional `labeling.py` implements `acquisition_function`; the platform supplies the current `input`, its `prediction` computed using previously acquired labels, the `labeled` evidence, and `context`. Return `True` to reveal one recorded response or `False` to skip the item permanently; its outcome is hidden until acquisition. Without this file, the platform selects labels randomly.

```
# labeling.py (optional)
def acquisition_function(input: list[dict],
                         prediction: float | None = None,
                         labeled: list | None = None,
                         context: dict | None = None) -> bool:
    """Decide whether to reveal the current candidate's outcome."""
    ...
```

`labeled` contains `[[subject, item], label]` entries across sampled subject–benchmark pairs. The dictionaries match the `input` structure, and `label` is the observed outcome: `1` for success or `0` otherwise. `context` includes `subject_id`, `benchmark_id`, `labels_acquired`, `labels_remaining`, `max_labels` (31), and `items_remaining` (including the current candidate). Counts apply to the current subject–benchmark pair; `labels_remaining` is the allowance left out of 31, not the distance to the next scoring budget.

### Evaluation.

#### Metric

ALC denotes the area under the learning curve, summarizing prediction error across label budgets. Brier ALC measures predictive performance for all submissions, with or without test-time adaptation; lower values indicate better performance. We measure Brier score at label budgets of 0, 1, 3, 7, 15, and 31, using the same fixed evaluation items and excluding acquisition items. Brier score is the mean squared difference between predicted probabilities and observed 0 or 1 outcomes. A score of 0 is perfect; always predicting 0.5 gives 0.25. The logarithmic budget axis places greater weight on predictive performance at small label budgets, reflecting the goal of accurate prediction from limited evaluation data. Using x=log⁡2(n+1)x = \\log\_2(n + 1)x=log2​(n+1) also allows the curve to include the zero-label condition and places the chosen budgets at equal intervals.

Illustrative predictor

Brier ALC ↓

0.140

Label budget, n · log₂(1 + n) spacing

Area under the curveNormalized ALC

Select a label budget to inspect its score

Budget 0: Brier 0.220.

Illustrative values showing improvement as labels are acquired.

##### Computing Brier ALC

Brier ALC is the normalized area under the learning curve, using x=log⁡2(n+1)x = \\log\_2(n + 1)x=log2​(n+1). Let BnB\_nBn​ denote the Brier score at label budget nnn. We calculate Brier ALC for each subject–benchmark pair, then average the pair scores equally. Every pair has the same weight, regardless of its number of responses.

Brier ALC=0.1 B0+0.2 B1+0.2 B3+0.2 B7+0.2 B15+0.1 B31\\text{Brier ALC} = 0.1\\,B\_0 + 0.2\\,B\_1 + 0.2\\,B\_3 + 0.2\\,B\_7 + 0.2\\,B\_{15} + 0.1\\,B\_{31}Brier ALC=0.1B0​+0.2B1​+0.2B3​+0.2B7​+0.2B15​+0.1B31​

Submissions without an acquisition function still run at all six budgets, using random label selection. The predictor receives the revealed labels and may adapt. Identical predictions across budgets give a flat curve, so Brier ALC equals Brier score.

How the weights are calculated+

1. Transform the six budgets into equally spaced positions 0, 1, 2, 3, 4, and 5.
2. Join adjacent Brier scores with straight lines and add the five trapezoid areas.
3. Divide by the total axis width of 5 to obtain the normalized area.

Each endpoint contributes to one trapezoid, giving it weight 0.1. Each interior point contributes to two, giving it weight 0.2. The weights sum to 1, keeping Brier ALC on the Brier scale.

#### Formative feedback

Formative feedback is generated automatically during method development and is available only to the submitter and organizers in Codabench under Logs → Scoring output. An ALC summary and per-budget tables report unique subject–item counts, Brier scores, and expected calibration error (ECE) for each evaluated subject–benchmark pair. Anonymous subject and benchmark IDs are randomly assigned and permanent across submissions and evaluation snapshots. Matching IDs always identify the same subject or benchmark.

Example formative feedback. Illustrative values; use the dropdown to view each table.

ALC summary

ViewALC summaryBudget 0Budget 1Budget 3Budget 7Budget 15Budget 31

| subject\_id | Benchmark | Number of Unique Subject-Item Pairs | Brier ALC ↓ | Calibration ECE ↓ |
|---|---|---|---|---|
| subject\_193804 | benchmark\_816205 | 73 | 0.157200 | 0.104300 |
| subject\_582731 | benchmark\_274926 | 46 | 0.140000 | 0.121000 |
| subject\_582731 | benchmark\_639418 | 62 | 0.215900 | 0.141000 |
| subject\_741962 | benchmark\_274926 | 117 | 0.184700 | 0.127100 |
| subject\_908517 | benchmark\_461073 | 181 | 0.117500 | 0.063900 |

Illustrative formative feedback: ALC summary

Number of Unique Subject-Item Pairs: The count represents distinct evaluated subject–item pairs and is fixed across budgets. Each pair is counted once, while all recorded responses contribute to Brier score and ECE. Brier score / Brier ALC ↓: Brier score measures prediction error at one budget; Brier ALC summarizes it across budgets using the weights above. Lower values indicate more accurate predictions. Calibration ECE ↓: ECE is the mean absolute gap between average predicted probabilities and observed success rates across 10 equal-width probability bins, weighted by response count. Summary ECE uses the ALC budget weights. Lower ECE indicates better calibration, although the estimate is sensitive to sample size and is reported only as a diagnostic.

#### Summative evaluation

The final summative evaluation combines automatic assessment on a common hidden test subset with manual grading of the submitted codebase and technical report. The automatic component measures predictive performance using Brier ALC. The organizers will review and rerun the released method to assess whether its reported results can be reproduced with reasonable effort, allowing for expected variation from sampling, randomized algorithms, and API outputs.

Technical report quality will be an important factor in selecting submissions for both oral and poster presentations. Selection also considers predictive performance, scientific rigor, and contributions to data curation and the competition community.

Technical reports will be assessed by the competition organizers using criteria adapted from the [NeurIPS 2026 Evaluations & Datasets reviewing guidelines](https://neurips.cc/Conferences/2026/EvaluationsDatasetsReviewerGuidelines). The primary emphasis will be on technical soundness, clarity, reproducibility, and the quality of experimental analysis. Methodological novelty will be a secondary consideration.

We do not currently plan to conduct formal peer review of technical reports.

## Resources.

### Data.

Candidate benchmarks are randomly assigned to separate training and test pools before curation. The [benchmark inventory](https://docs.google.com/spreadsheets/d/1T69QyIZJRWX3d-Tmuh_lcFwtpqv9ytcLxfLWKn6gopI/edit?gid=0#gid=0) and its [construction methodology](https://docs.google.com/document/d/1LdUEiUG5tLhDdnIAtm4YKSf4w3LFb2HmI_ahiiuoRLU/edit?tab=t.h6mifjz8cok) describe the candidate collection, while the [data curation guidelines](https://docs.google.com/document/d/1LdUEiUG5tLhDdnIAtm4YKSf4w3LFb2HmI_ahiiuoRLU/edit?tab=t.jcvir9i0usja) explain how participants can contribute. Verified training data, including participant contributions, are released through [measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db), while test data remain private. All data sources listed in the benchmark inventory are permitted for competition-specific training, development, and curation. Participants must not use data sources outside that inventory for these purposes. The collections may grow during the competition, while pool assignments and the sampling design remain fixed. Each formative evaluation uses a newly sampled subset of the hidden test data, capped at 1,000 unique subject–item pairs across the acquisition and evaluation pools. All label budgets within a submission use the same evaluation responses, and the automatic component of summative evaluation uses a common test subset for all teams.

### Baselines.

The public [PAIEC baseline repository](https://github.com/aims-foundations/paiec_baseline) provides reference predictors and acquisition functions for participants to use and adapt. Implementation details, setup instructions, and submission packaging are maintained alongside the code.

## Q&A.

### General

Who can participate?+

Participation is open to individuals regardless of affiliation. [Eligibility and registration](https://aims.stanford.edu/competition#logistics)

Why is the submission code rather than a predictions file?+

The platform runs the submitted program on hidden test inputs and generates the predictions file, keeping the test data private. [Predictor interface](https://aims.stanford.edu/competition#interface)

### Technical

What programming languages are supported?+

The platform supports Python with preinstalled dependencies. Extra package installation and `trust_remote_code` are disabled by default and controlled by organizer policy. [Package requirements](https://aims.stanford.edu/competition#submission-package)

What does `predict()` receive at runtime?+

`input` contains `[subject, item]`; `labeled` contains outcomes revealed at the current budget. [Complete input field reference](https://aims.stanford.edu/competition#predictor-fields)

What must `predict()` return?+

The prediction function must return a finite probability in `[0, 1]`. [Return-value requirements](https://aims.stanford.edu/competition#predictor-output)

How do the hidden test data relate to the training release?+

Benchmarks are randomly assigned to separate training and test pools before curation. [Data and curation](https://aims.stanford.edu/competition#data)

Which subjects are included in my formative feedback?+

The platform selects the subjects. Stable anonymous IDs identify recurring subjects and benchmarks. [Formative feedback and IDs](https://aims.stanford.edu/competition#formative-feedback)

Can I use pretrained models?+

Participants may use pretrained models and must disclose them in the technical report. Any competition-specific training or fine-tuning must use only data sources listed in the benchmark inventory. [Model packaging](https://aims.stanford.edu/competition#submission-package)

Can I use external APIs?+

Yes. Network access during scoring is limited to inference APIs from Anthropic, OpenAI, Google, xAI, DeepSeek, Mistral, Together AI, and Groq. Web search, browsing, and grounding are prohibited, including when offered through an approved provider's API. [Runtime rules](https://aims.stanford.edu/competition#runtime-rules)

What are the compute limits?+

Each run has 8 hours, including setup. Up to 5 model repositories are allowed: 300B parameters and 1,000 GB per repository; 1,024 GB combined. GPU allocation is automatic. Supply your own development compute and API keys; models cannot be downloaded at runtime. [Model packaging](https://aims.stanford.edu/competition#submission-package)

### Submissions

How is predictive performance scored?+

Predictive performance is measured by averaging Brier ALC equally across subject–benchmark pairs. Lower scores indicate better performance; ECE is reported as a diagnostic. [Evaluation metric and curve](https://aims.stanford.edu/competition#evaluation-metric)

Who can see the submission feedback?+

Individual feedback is private to the submitter and organizers. Aggregate leaderboard visibility is announced separately. [Formative feedback](https://aims.stanford.edu/competition#formative-feedback)

Why is the formative sample redrawn for every submission?+

Resampling provides feedback across the hidden test pool while limiting the evaluation workload per submission. The final evaluation uses a common test subset for all teams. [Sampling rules](https://aims.stanford.edu/competition#data-sampling)

What do the failure codes mean?+

The local ZIP check and smoke test can help diagnose submission failures. [Troubleshooting and failure codes](https://aims.stanford.edu/competition#submission-troubleshooting)

## Organizers.

![Sang Truong](https://aims.stanford.edu/_next/image?url=%2Fpeople%2Fsang-truong.jpg&w=3840&q=75)

### [Sang Truong](https://cs.stanford.edu/~sttruong)

Stanford University

![Wale Salaudeen](https://aims.stanford.edu/_next/image?url=%2Fpeople%2Folawale-salaudeen.jpeg&w=3840&q=75)

### [Wale Salaudeen](https://olawalesalaudeen.com)

Microsoft Research / UIUC

![Nhi Truong](https://aims.stanford.edu/_next/image?url=%2Fpeople%2Fnhi-truong.jpg&w=3840&q=75)

### [Nhi Truong](https://cs.stanford.edu/~nntruong/)

Stanford University

![Serena Wang](https://aims.stanford.edu/_next/image?url=%2Fpeople%2Fserena-wang.jpg&w=3840&q=75)

### [Serena Wang](https://serenalwang.com/)

UBC, Google Research

![Angelina Wang](https://aims.stanford.edu/_next/image?url=%2Fpeople%2Fangelina-wang.jpg&w=3840&q=75)

### [Angelina Wang](https://angelina-wang.github.io/)

Cornell University

![Luke Guerdan](https://aims.stanford.edu/_next/image?url=%2Fpeople%2Fluke-guerdan.jpg&w=3840&q=75)

### [Luke Guerdan](https://lukeguerdan.com)

Carnegie Mellon University

![Ziang Xiao](https://aims.stanford.edu/_next/image?url=%2Fpeople%2Fziang-xiao.jpg&w=3840&q=75)

### [Ziang Xiao](https://www.ziangxiao.com/)

Johns Hopkins University

![Xing Xie](https://aims.stanford.edu/_next/image?url=%2Fpeople%2Fxing-xie.jpg&w=3840&q=75)

### [Xing Xie](https://www.microsoft.com/en-us/research/people/xingx/)

Microsoft Research

![Nick Haber](https://aims.stanford.edu/_next/image?url=%2Fpeople%2Fnick-haber.jpg&w=3840&q=75)

### [Nick Haber](https://ed.stanford.edu/faculty/nhaber)

Stanford University

![Sanmi Koyejo](https://aims.stanford.edu/_next/image?url=%2Fpeople%2Fsanmi-koyejo.jpg&w=3840&q=75)

### [Sanmi Koyejo](https://cs.stanford.edu/~sanmi)

Stanford University