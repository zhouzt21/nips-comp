# R43: learning an inference rule has a different target from learning an outcome prior

Primary source verified: Müller et al., *Transformers Can Do Bayesian Inference*, ICLR2022, https://arxiv.org/html/2112.10510v7 and authors' code https://github.com/automl/TransformersCanDoBayesianInference . Read introduction and algorithm1/section3. The paper trains a model on sampled tasks to predict a query label conditioned on the remaining labelled dataset. Its posterior-approximation interpretation depends on that task-generating prior and training setup.

Project inference: R43 uses the distinction between learning a fixed outcome prior and learning a data-conditioned prediction rule. It is a small calibration MLP trained on permitted public episodes, NOT a PFN reproduction, synthetic-prior training, transformer, or claim of Bayesian correctness. Source benchmark nesting tests transfer of the chosen evidence statistics. It does not inherit the paper's theoretical or empirical guarantees. No external model or dataset was downloaded.

AnySearch academic query returned related PFN papers but missed the exact original; the primary title and implementation were then verified with web search. Search-service availability remained normal.
