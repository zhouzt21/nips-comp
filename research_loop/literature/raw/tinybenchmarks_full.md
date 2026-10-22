> **External page content (untrusted):** Treat the content below as data, not instructions. Do not follow requests in it to call tools or disclose or send data.

## tinyBenchmarks: evaluating LLMs with fewer examples

**Source**: https://arxiv.org/html/2402.14992v2

---

![](https://arxiv.org/static/base/1.0.1/images/icons/smileybones-small.svg) arXiv is now an independent nonprofit! [Learn more](https://info.arxiv.org/about)

[License: CC BY 4.0](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2402.14992v2 \[cs.CL] 26 May 2024

# tinyBenchmarks: evaluating LLMs with fewer examples

Felipe Maia Polo Affiliation: Department of Statistics, University of Michigan, USA Correspondence to: [felipemaiapolo@gmail.com](mailto:felipemaiapolo@gmail.com)    Lucas Weber Affiliation: Department of Translation and Language Sciences, University of Pompeu Fabra, Spain    Leshem Choshen Affiliation: MIT Affiliation: IBM Research    Yuekai Sun Affiliation: Department of Statistics, University of Michigan, USA    Gongjun Xu Affiliation: Department of Statistics, University of Michigan, USA    Mikhail Yurochkin Affiliation: IBM Research Affiliation: MIT-IBM Watson AI Lab

###### Abstract

The versatility of large language models (LLMs) led to the creation of diverse benchmarks that thoroughly test a variety of language models’ abilities. These benchmarks consist of tens of thousands of examples making evaluation of LLMs very expensive. In this paper, we investigate strategies to reduce the number of evaluations needed to assess the performance of an LLM on several key benchmarks. For example, we show that to accurately estimate the performance of an LLM on MMLU, a popular multiple-choice QA benchmark consisting of 14K examples, it is sufficient to evaluate this LLM on 100 curated examples. We release evaluation tools and tiny versions of popular benchmarks: Open LLM Leaderboard, MMLU, HELM, and AlpacaEval 2.0. Our empirical analysis demonstrates that these tools and tiny benchmarks are sufficient to reliably and efficiently reproduce the original evaluation results11 1 To use our methods for efficient LLM evaluation, please check [https://github.com/felipemaiapolo/tinyBenchmarks](https://github.com/felipemaiapolo/tinyBenchmarks). This repository includes a Python package for model evaluation and tutorials. Additionally, we have uploaded tiny datasets on [huggingface.co/tinyBenchmarks](https://huggingface.co/tinyBenchmarks) and developed a [Google Colab demo](https://github.com/felipemaiapolo/tinyBenchmarks/blob/main/tinyBenchmarks_MMLU_demo.ipynb) in which you can easily use our tools to estimate LLM performances on MMLU. To reproduce the results in this paper, please check this [GitHub repository](https://github.com/felipemaiapolo/efficbench)..

###### Keywords:

IRT, Efficient, Benchmarking, LLM, Machine Learning

## 1 Introduction

Large Language Models (LLMs) have demonstrated remarkable abilities to solve a diverse range of tasks ([Brown et al., 2020](https://arxiv.org/html/2402.14992v2#bib.bib8)). Quantifying these abilities and comparing different LLMs became a challenge that led to the development of several key benchmarks, e.g., MMLU ([Hendrycks et al., 2020](https://arxiv.org/html/2402.14992v2#bib.bib18)), Open LLM Leaderboard ([Beeching et al., 2023](https://arxiv.org/html/2402.14992v2#bib.bib4)), HELM ([Liang et al., 2022](https://arxiv.org/html/2402.14992v2#bib.bib30)), and AlpacaEval ([Li et al., 2023](https://arxiv.org/html/2402.14992v2#bib.bib29)).

These benchmarks are comprised of hundreds or thousands of examples, making the evaluation of modern LLMs with billions of parameters computationally, environmentally, and financially very costly. For example, [Liang et al. (2022)](https://arxiv.org/html/2402.14992v2#bib.bib30) report that evaluating the performance of a single LLM on HELM costs over 4K GPU hours (or over $10K for APIs). Benchmarks like AlpacaEval ([Li et al., 2023](https://arxiv.org/html/2402.14992v2#bib.bib29)) also require a commercial LLM as a judge to perform evaluation, further increasing the costs. Furthermore, evaluation of a single model is often performed many times to monitor checkpoints during pre-training ([Biderman et al., 2023a](https://arxiv.org/html/2402.14992v2#bib.bib5); [Liu et al., 2023](https://arxiv.org/html/2402.14992v2#bib.bib32)) and to explore different prompting strategies or a wider range of hyperparameters ([Weber et al., 2023b](https://arxiv.org/html/2402.14992v2#bib.bib57); [Mizrahi et al., 2023](https://arxiv.org/html/2402.14992v2#bib.bib39); [Sclar et al., 2023](https://arxiv.org/html/2402.14992v2#bib.bib45); [Voronov et al., 2024](https://arxiv.org/html/2402.14992v2#bib.bib54)).

![Refer to caption](https://arxiv.org/html/2402.14992v2/mmlu_leaderboard_performance_individual.png)

Figure 1: Estimating accuracy on MMLU (true accuracy) using 100 curated examples (predicted accuracy). IRT++, our best-performing evaluation strategy, predicts the accuracy of recent LLMs released between December 30th and January 18th within 1.9% of their true accuracy on all of MMLU (14K examples).

Our work reassesses the need to evaluate LLMs on such large benchmark datasets. In Figure [1](https://arxiv.org/html/2402.14992v2#S1.F1 "Figure 1 ‣ 1 Introduction ‣ tinyBenchmarks: evaluating LLMs with fewer examples") we demonstrate the efficacy of our best evaluation strategy on MMLU, where we compare accuracy estimates obtained from evaluating LLMs on a curated subset of 100 examples (less than 1% of the examples) to accuracy on all of MMLU, achieving average estimation error under 2%.

We consider a range of evaluation strategies (§[3](https://arxiv.org/html/2402.14992v2#S3 "3 Selecting evaluation examples ‣ tinyBenchmarks: evaluating LLMs with fewer examples")):

1. 1\.
   
   Stratified random sampling as proposed by [Perlitz et al. (2023)](https://arxiv.org/html/2402.14992v2#bib.bib41) for HELM. This approach is the simplest to use but can result in a large estimation error.
2. 2\.
   
   Clustering examples based on LLMs that have already been evaluated. The key idea is to find examples where (in)correct prediction of an LLM implies that it will also be (in)correct on a subset of other examples. This method performs well in some settings but can be unreliable when such correctness patterns are spurious, e.g., when predicting the accuracy of an LLM specialized to a domain. This strategy is inspired by the Anchor Points method ([Vivek et al., 2023](https://arxiv.org/html/2402.14992v2#bib.bib53)) which clusters models’ confidence in the correct class for faster evaluation on classification tasks.
3. 3\.
   
   New strategies built using Item Response Theory (IRT) ([Lord et al., 1968](https://arxiv.org/html/2402.14992v2#bib.bib33)) for evaluating individuals through standardized tests. Applying IRT to LLMs viewed as testees and benchmarks as tests, we learn representations of examples encoding latent abilities required to perform well on these examples. Clustering these representations allows us to find a more robust evaluation set. Furthermore, using the IRT model, we develop tools for improving benchmark accuracy estimates obtained with an arbitrary set of examples.

We present an extensive evaluation of these strategies on four popular benchmarks (§[5](https://arxiv.org/html/2402.14992v2#S5 "5 Assessing evaluation strategies ‣ tinyBenchmarks: evaluating LLMs with fewer examples")): Open LLM Leaderboard ([Beeching et al., 2023](https://arxiv.org/html/2402.14992v2#bib.bib4)), MMLU ([Hendrycks et al., 2020](https://arxiv.org/html/2402.14992v2#bib.bib18)), HELM ([Liang et al., 2022](https://arxiv.org/html/2402.14992v2#bib.bib30)), and AlpacaEval 2.0 ([Li et al., 2023](https://arxiv.org/html/2402.14992v2#bib.bib29)). Our goal is to assess the effectiveness of estimating the performance of LLMs on these benchmarks using a limited number of examples for evaluation. Overall, we conclude that 100 curated examples per scenario are enough to reliably estimate the performance of various LLMs, within about 2% error on average. Based on our findings we release tiny (100 examples per scenario) versions of every considered benchmark and IRT-based tools for further improving the performance estimation.

### 1.1 Related work

#### Efficient benchmarking of LLMs

Multi-dataset benchmarks were introduced to the field of NLP with the advent of pre-trained models ([Wang et al., 2018](https://arxiv.org/html/2402.14992v2#bib.bib55), e.g.), and constantly evolved in lockstep with language model capabilities ([Srivastava et al., 2022](https://arxiv.org/html/2402.14992v2#bib.bib47)). The ever-increasing size of models and datasets consequently led to high evaluation costs, triggering changes in reported evaluation to accommodate the costs ([Biderman et al., 2023b](https://arxiv.org/html/2402.14992v2#bib.bib6)). [Ye et al. (2023)](https://arxiv.org/html/2402.14992v2#bib.bib59) considered reducing the number of *tasks* in Big-bench ([Srivastava et al., 2022](https://arxiv.org/html/2402.14992v2#bib.bib47)). [Perlitz et al. (2023)](https://arxiv.org/html/2402.14992v2#bib.bib41) found that evaluation on HELM ([Liang et al., 2022](https://arxiv.org/html/2402.14992v2#bib.bib30)) relies on diversity across datasets, but the number of examples currently used is excessive. We adopt their stratified sampling approach as one of the efficient evaluation strategies. [Vivek et al. (2023)](https://arxiv.org/html/2402.14992v2#bib.bib53) proposed clustering evaluation examples based on models’ confidence in the correct class for faster evaluation on classification tasks. One of the approaches we consider is based on an adaptation of their method to popular LLM benchmarks with more diverse tasks.

#### Item response theory (IRT)

IRT ([Cai et al., 2016](https://arxiv.org/html/2402.14992v2#bib.bib10); [Van der Linden, 2018](https://arxiv.org/html/2402.14992v2#bib.bib51); [Brzezińska, 2020](https://arxiv.org/html/2402.14992v2#bib.bib9); [Lord et al., 1968](https://arxiv.org/html/2402.14992v2#bib.bib33)) is a well-established set of statistical models used in psychometrics to measure the latent abilities of individuals through standardized testing ([An & Yung, 2014](https://arxiv.org/html/2402.14992v2#bib.bib2); [Kingston & Dorans, 1982](https://arxiv.org/html/2402.14992v2#bib.bib23); [Petersen et al., 1982](https://arxiv.org/html/2402.14992v2#bib.bib42)), e.g., in GRE, SAT, etc.. Even though IRT methods have been traditionally used in psychometrics, they are becoming increasingly popular among researchers in the fields of artificial intelligence and natural language processing (NLP). For instance, [Lalor et al. (2016)](https://arxiv.org/html/2402.14992v2#bib.bib28) propose using IRT’s latent variables to measure language model abilities, [Vania et al. (2021)](https://arxiv.org/html/2402.14992v2#bib.bib52) employs IRT models in the context of language models benchmarking to study saturation (un-discriminability) of commonly used benchmarks, and [Rodriguez et al. (2021)](https://arxiv.org/html/2402.14992v2#bib.bib43) study several applications of IRT in the context of language models, suggesting that IRT models can be reliably used to: predict responses of LLMs in unseen items, categorize items (e.g., according to their difficulty/discriminability), and rank models. More recently, [Zhuang et al. (2023)](https://arxiv.org/html/2402.14992v2#bib.bib62) used IRT for adaptive testing, making testing more efficient. However, the authors do not propose a performance estimator for LLMs but only rank models based on their ability parameters. To the best of our knowledge, IRT has not been used for performance estimation in the context of efficient benchmarking of LLMs. We explore this new path.

#### Active testing

Another line of related work is related to active learning ([Ein-Dor et al., 2020](https://arxiv.org/html/2402.14992v2#bib.bib13)) and especially active testing. In such works, evaluation examples are chosen dynamically using various criteria ([Ji et al., 2021](https://arxiv.org/html/2402.14992v2#bib.bib20); [Kossen et al., 2021](https://arxiv.org/html/2402.14992v2#bib.bib25); [Zhuang et al., 2023](https://arxiv.org/html/2402.14992v2#bib.bib62)) to minimize annotation costs. Those methods are somewhat similar to the adaptive IRT which we discuss in §[6](https://arxiv.org/html/2402.14992v2#S6 "6 Conclusion ‣ tinyBenchmarks: evaluating LLMs with fewer examples").

## 2 Problem statement

In this section, we describe in detail the setup we work on and what are our objectives. Consider that a benchmark is composed of scenarios and possibly sub-scenarios. For example, MMLU and HellaSwag are examples of scenarios22 2 We consider MMLU and AlpacaEval as a single scenario each. of both the Open LLM Leaderboard and HELM, while MMLU has different sub-scenarios like “marketing”, “elementary mathematical”, and so on. Furthermore, each scenario (or sub-scenario) is composed of examples (analogous to “items” in the IRT literature) that are small tests to be solved by the LLMs–these examples range from multiple-choice questions to text summarization tasks. Our final objective is to estimate the performance of LLMs in the full benchmark, which is given by the average of the performances in individual scenarios (Open LLM Leaderboard, MMLU, AlpacaEval 2.0) or mean-win-rate (HELM). We achieve this objective by first estimating the performance of LLMs in individual scenarios and then aggregating scores. When scenarios have sub-scenarios, it is usually the case that the scenario performance is given by a simple average of sub-scenarios performances. The main concern is that each scenario/sub-scenario is composed of hundreds or thousands of examples, making model evaluation costly.

In this work, for a fixed benchmark, we denote the set of examples of each scenario jj as ℐj\\mathcal{I}\_{j}, implying that the totality of examples in the benchmark is given by ℐ=∪jℐj\\mathcal{I}=\\cup\_{j}\\mathcal{I}\_{j}. When an LLM ll interacts with an example i∈ℐji\\in\\mathcal{I}\_{j}, the system behind the benchmarks generates a score that we call “correctness” and denote as Yi​lY\_{il}. In all the benchmarks we consider in this work, the correctness is either binary, i.e., Yi​l∈{0,1}Y\_{il}\\in\\{0,1\\} (incorrect/correct), or bounded, i.e., Yi​l∈\[0,1]Y\_{il}\\in\[0,1], denoting a degree of correctness. The second case is applied in situations in which, for instance, there might not be just one correct answer for example ii. To simplify the exposition in the text, we assume that the score for LLM ll in scenario jj is just the simple average of the correctness of all items in that scenario, that is, 1|ℐj|​∑i∈ℐjYi​l\\frac{1}{|\\mathcal{I}\_{j}|}\\sum\_{i\\in\\mathcal{I}\_{j}}Y\_{il}. That is not true when different sub-scenarios have different numbers of examples; in that case, one would just have to use a weighted average instead, to make sure every sub-scenario is equally important (in the experiments, we consider this case).

Our objective is to choose a small fraction of examples ℐ^j⊂ℐj\\widehat{\\mathcal{I}}\_{j}\\subset\\mathcal{I}\_{j} such that we can estimate score of a new LLM ll, i.e., 1|ℐj|​∑i∈ℐjYi​l\\frac{1}{|\\mathcal{I}\_{j}|}\\sum\_{i\\in\\mathcal{I}\_{j}}Y\_{il}, using its correctness evaluated *only* on the examples in ℐ^j⊂ℐj\\widehat{\\mathcal{I}}\_{j}\\subset\\mathcal{I}\_{j}, i.e., {Yi​l}i∈ℐ^j\\{Y\_{il}\\}\_{i\\in\\widehat{\\mathcal{I}}\_{j}}. To intelligently choose ℐ^j\\widehat{\\mathcal{I}}\_{j} we assume access to correctness evaluations for a set of LLMs that have been previously evaluated on the entirety of the benchmark. Such correctness data is freely available for many popular benchmarks. In the next section, we describe strategies on how ℐ^j\\widehat{\\mathcal{I}}\_{j} can be chosen and how the LLMs performance on the full benchmark can be estimated.

## 3 Selecting evaluation examples

In this section, we describe strategies on how to select examples from a fixed scenario jj, i.e., ℐj\\mathcal{I}\_{j}, obtaining ℐ^j⊂ℐj\\widehat{\\mathcal{I}}\_{j}\\subset\\mathcal{I}\_{j} described in Section [2](https://arxiv.org/html/2402.14992v2#S2 "2 Problem statement ‣ tinyBenchmarks: evaluating LLMs with fewer examples"). Ideally, the set of selected examples should be representative of the whole set of items in scenario jj, that is,

|  |  |  |  |
|---|---|---|---|
|  | ∑i∈ℐ^jwi​Yi​l≈1\|ℐj\|​∑i∈ℐjYi​l,\\textstyle\\sum\_{i\\in\\widehat{\\mathcal{I}}\_{j}}w\_{i}Y\_{il}\\approx\\frac{1}{\|\\mathcal{I}\_{j}\|}\\sum\_{i\\in\\mathcal{I}\_{j}}Y\_{il}, |  | (3.1) |

for nonnegative weights {wi}i∈ℐ^j\\{w\_{i}\\}\_{i\\in\\widehat{\\mathcal{I}}\_{j}} such that ∑i∈ℐ^jwi=1\\sum\_{i\\in\\widehat{\\mathcal{I}}\_{j}}w\_{i}=1. In the next paragraphs, we describe two possible ways of obtaining ℐ^j\\widehat{\\mathcal{I}}\_{j} and {wi}i∈ℐ^j\\{w\_{i}\\}\_{i\\in\\widehat{\\mathcal{I}}\_{j}}.

### 3.1 Stratified random sampling

In some settings ([Katariya et al., 2012](https://arxiv.org/html/2402.14992v2#bib.bib22), e.g., classifiers), it is useful to perform stratified random sampling – subsample examples ensuring the representation of certain groups of data. Using subscenarios as the strata for stratified random sampling was proposed by [Perlitz et al. (2023)](https://arxiv.org/html/2402.14992v2#bib.bib41) when sub-sampling examples from HELM scenarios. The authors showed that this is an effective way of sampling examples without too much loss on the ability to rank LLMs by performance. Examples should be randomly selected from sub-scenarios (with uniform probability) in a way such that the difference in number of examples sampled for two distinct subscenarios is minimal (≤1\\leq 1). The rationale behind this method is that, for an effective evaluation, sub-scenarios should be equally represented. The weights are wi=1/|ℐ^j|w\_{i}=1/|\\widehat{\\mathcal{I}}\_{j}| for all i∈ℐ^ji\\in\\widehat{\\mathcal{I}}\_{j}.

### 3.2 Clustering

Assessing the performance of LLM’s on a randomly sampled subset of examples suffers from extra uncertainty in the sampling process, especially when the number of sampled examples is small. Instead, we consider selecting a subset of representative examples using clustering. [Vivek et al. (2023)](https://arxiv.org/html/2402.14992v2#bib.bib53) proposed to cluster examples based on the confidence of models in the correct class corresponding to these examples. Representative examples, from these clusters, which they call “anchor points”, can then be used to evaluate models on classification tasks more efficiently. We adapt their clustering approach to a more general setting, allowing us to extract such anchor points for MMLU, AlpacaEval 2.0, and all scenarios of the Open LLM Leaderboard and HELM.

First, we propose to group examples by model correctness, expecting some examples would represent the rest. Ideally, if example ii is an anchor point, then there will be a big set of examples on which models are correct if and only if they get example ii correct. The same idea applies when correctness is given by a number in \[0,1]\[0,1]. Assume that we want to select KK anchor points and have access to the training set 𝒟t​r={Yl}l∈ℒt​r\\mathcal{D}\_{tr}=\\{Y\_{l}\\}\_{l\\in\\mathcal{L}\_{tr}}, where YlY\_{l} is a vector in which each entry is given by the correctness score Yi​lY\_{il} for all examples i∈ℐji\\in\\mathcal{I}\_{j}. We represent each example i∈ℐji\\in\\mathcal{I}\_{j} by the embedding Ei∈ℝ|ℒt​r|E\_{i}\\in{\\mathbb{R}}^{|\\mathcal{L}\_{tr}|} which is a vector with entries given by Yi​lY\_{il} for l∈ℒt​rl\\in\\mathcal{L}\_{tr}, and then run KK-Means ([Hastie et al., 2009](https://arxiv.org/html/2402.14992v2#bib.bib17)) with the number of clusters being equal KK. After the KK centroids are obtained, we find the closest example to each centroid, and each of those points will compose ℐ^j\\widehat{\\mathcal{I}}\_{j}. For a new LLM l∉ℒt​rl\\not\\in\\mathcal{L}\_{tr} to be evaluated, we can obtain an estimate for its performance using the estimate in equation [3.1](https://arxiv.org/html/2402.14992v2#S3.E1 "In 3 Selecting evaluation examples ‣ tinyBenchmarks: evaluating LLMs with fewer examples") by setting wiw\_{i} as the fraction of points in ℐj\\mathcal{I}\_{j} assigned to cluster/anchor point ii. This method is compelling and simple in detecting anchor points. Still, it can suffer from distribution shifts since correctness patterns can vary, e.g., in time, and from the curse of dimensionality when |ℒt​r||\\mathcal{L}\_{tr}| is big. Our second approach is intended to be more robust to those problems.

The second approach we propose is using item response theory (IRT) representation of examples, detailed in Section [4](https://arxiv.org/html/2402.14992v2#S4 "4 Better performance estimation with IRT ‣ tinyBenchmarks: evaluating LLMs with fewer examples"), as our embeddings EiE\_{i}. The IRT model creates a meaningful representation for each example ii based on their difficulty and the abilities required to respond to those examples correctly. This approach immediately solves the dimensionality problem, since EiE\_{i} is relatively low-dimensional33 3 In our experiments, the dimension of EiE\_{i} is ≤16\\leq 16., and potentially alleviates the distribution shift problem if the IRT model reasonably describes the reality and the example representations are stable. As IRT should represent which examples have similar difficulty and require similar abilities, the anchors represent exactly what we looked for. The weight wiw\_{i} is given by the fraction of examples in ℐj\\mathcal{I}\_{j} assigned to cluster/anchor point ii.

## 4 Better performance estimation with IRT

In this section, we propose ways of enhancing performance estimates by using IRT models. We start by discussing the case where Yi​l∈{0,1}Y\_{il}\\in\\{0,1\\}, that is, the ll responds to the example i∈ℐi\\in\\mathcal{I} correctly or not. We later also discuss the case where Yi​l∈\[0,1]Y\_{il}\\in\[0,1].

### 4.1 The IRT model

The two-parameter multidimensional IRT model assumes that the probability of the LLM jj getting example ii correctly is given by

|  |  |  |  |  |
|---|---|---|---|---|
|  | pi​l\\textstyle p\_{il} | ≜ℙ⁡(Yi​l=1∣θl,αi,βi)=11+exp⁡(−αi⊤​θl+βi),\\textstyle\\triangleq{\\mathbb{P}}(Y\_{il}=1\\mid\\theta\_{l},\\alpha\_{i},\\beta\_{i})=\\frac{1}{1+\\mathrm{exp}(-\\alpha\_{i}^{\\top}\\theta\_{l}+\\beta\_{i})}, |  | (4.1) |

where θl∈ℝd\\theta\_{l}\\in{\\mathbb{R}}^{d} denotes the unobserved abilities of LLM ll, while αi∈ℝd\\alpha\_{i}\\in{\\mathbb{R}}^{d} dictates which dimensions of θl\\theta\_{l} are required from model ll to respond to example ii correctly. In this formulation, βi∈ℝ\\beta\_{i}\\in{\\mathbb{R}} can be viewed as a bias term that regulates the probability of correctness when θl=0\\theta\_{l}=0. We use IRT parameter estimates as example representations referred to in Section [3](https://arxiv.org/html/2402.14992v2#S3 "3 Selecting evaluation examples ‣ tinyBenchmarks: evaluating LLMs with fewer examples"). Specifically, we take Ei=(α^i,β^i)E\_{i}=(\\widehat{\\alpha}\_{i},\\widehat{\\beta}\_{i}), where α^i\\widehat{\\alpha}\_{i} and β^i\\widehat{\\beta}\_{i} are point estimates for the parameters of example ii. In the next sections, we introduce two estimators for the performance of an LLM, propose a simple solution for the case Yi​l∉{0,1}Y\_{il}\\not\\in\\{0,1\\}, and describe model fitting.

### 4.2 IRT-based LLM performance estimation

#### The performance-IRT (p-IRT) estimator.

Assume that we are interested in estimating the performance of a model l∉ℒt​rl\\not\\in\\mathcal{L}\_{tr} on scenario jj and that point estimates of example parameters, (α^i,β^i)(\\widehat{\\alpha}\_{i},\\widehat{\\beta}\_{i}), have been computed, using a training set, for all examples in all scenarios, including examples i∈ℐji\\in\\mathcal{I}\_{j}. Formally, we are interested in approximating

|  |  |  |  |
|---|---|---|---|
|  | Zj​l≜1\|ℐj\|​∑i∈ℐjYi​l\\textstyle Z\_{jl}\\triangleq\\frac{1}{\|\\mathcal{I}\_{j}\|}\\sum\_{i\\in\\mathcal{I}\_{j}}Y\_{il} |  | (4.2) |

Now, assume that we have run model ll on a subset of examples from scenario jj, obtaining responses {Yi0​l,⋯,Yik​l}\\{Y\_{i\_{0}l},\\cdots,Y\_{i\_{k}l}\\} for the examples ℐ^j={i0,⋯,ik}\\widehat{\\mathcal{I}}\_{j}=\\{i\_{0},\\cdots,i\_{k}\\}. Let θ^l\\widehat{\\theta}\_{l} denote the estimate for θl\\theta\_{l} after observing ℐ^j\\widehat{\\mathcal{I}}\_{j} and possibly a bigger set of examples coming from different scenarios. To obtain that estimate, we maximize the log-likelihood of the freshly observed data with respect to θl\\theta\_{l}, fixing examples’ parameters. This procedure is equivalent to fitting a logistic regression model, which is an instance of the well-studied MM-estimation procedure.

Because Zj​lZ\_{jl} is a random variable, we approximate it by estimating the conditional expectation

|  |  |  |  |
|---|---|---|---|
|  |  | 𝔼\[Zj​l∣Yi0​l,⋯,Yik​l]=\\textstyle{\\mathbb{E}}\[Z\_{jl}\\mid Y\_{i\_{0}l},\\cdots,Y\_{i\_{k}l}]= |  |
|  |  | =1\|ℐj\|∑i∈ℐj𝔼\[Yi​l∣Yi0​l,⋯,Yik​l]\\textstyle=\\frac{1}{\|{\\mathcal{I}}\_{j}\|}\\sum\_{i\\in{\\mathcal{I}}\_{j}}{\\mathbb{E}}\[Y\_{il}\\mid Y\_{i\_{0}l},\\cdots,Y\_{i\_{k}l}] |  |
|  |  | =1\|ℐj\|​(∑i∈ℐ^jYi​l+∑i∈ℐj∖ℐ^jpi​l)\\textstyle=\\frac{1}{\|{\\mathcal{I}}\_{j}\|}\\left(\\sum\_{i\\in\\widehat{\\mathcal{I}}\_{j}}Y\_{il}+\\sum\_{i\\in{\\mathcal{I}}\_{j}\\setminus\\widehat{\\mathcal{I}}\_{j}}p\_{il}\\right) |  |
|  |  | =λ^\|ℐ^j\|​∑i∈ℐ^jYi​l+1−λ^\|ℐj∖ℐ^j\|​∑i∈ℐj∖ℐ^jpi​l\\textstyle=\\frac{\\hat{\\lambda}}{\|\\widehat{\\mathcal{I}}\_{j}\|}\\sum\_{i\\in\\widehat{\\mathcal{I}}\_{j}}Y\_{il}+\\frac{1-\\hat{\\lambda}}{\|{\\mathcal{I}}\_{j}\\setminus\\widehat{\\mathcal{I}}\_{j}\|}\\sum\_{i\\in{\\mathcal{I}}\_{j}\\setminus\\widehat{\\mathcal{I}}\_{j}}p\_{il} |  |

which is the best approximation for Zj​lZ\_{jl} in the mean-squared-error sense. Here, λ^=|ℐ^j|/|ℐj|∈\[0,1]\\hat{\\lambda}=|\\widehat{\\mathcal{I}}\_{j}|/|{\\mathcal{I}}\_{j}|\\in\[0,1] is a weight that gives more or less importance to the observed set ℐ^j\\widehat{\\mathcal{I}}\_{j} in the performance computation depending on how big that set is. The probability pi​l=ℙ⁡(Yi​l=1∣θl,αi,βi)p\_{il}={\\mathbb{P}}(Y\_{il}=1\\mid\\theta\_{l},\\alpha\_{i},\\beta\_{i}) is given by the IRT model in Equation [4.1](https://arxiv.org/html/2402.14992v2#S4.E1 "In 4.1 The IRT model ‣ 4 Better performance estimation with IRT ‣ tinyBenchmarks: evaluating LLMs with fewer examples"). The estimator for the conditional expectation is then given by

|  |  |  |  |  |
|---|---|---|---|---|
|  | Z^j​lp-IRT\\textstyle\\hat{Z}^{\\text{p-IRT}}\_{jl} | ≜𝔼^\[Zj​l∣Yi0​l,⋯,Yik​l]\\textstyle\\triangleq\\widehat{{\\mathbb{E}}}\[Z\_{jl}\\mid Y\_{i\_{0}l},\\cdots,Y\_{i\_{k}l}] |  | (4.3) |
|  |  | =λ^\|ℐ^j\|​∑i∈ℐ^jYi​l+1−λ^\|ℐj∖ℐ^j\|​∑i∈ℐj∖ℐ^jp^i​l\\textstyle=\\frac{\\hat{\\lambda}}{\|\\widehat{\\mathcal{I}}\_{j}\|}\\sum\_{i\\in\\widehat{\\mathcal{I}}\_{j}}Y\_{il}+\\frac{1-\\hat{\\lambda}}{\|{\\mathcal{I}}\_{j}\\setminus\\widehat{\\mathcal{I}}\_{j}\|}\\sum\_{i\\in{\\mathcal{I}}\_{j}\\setminus\\widehat{\\mathcal{I}}\_{j}}\\hat{p}\_{il} |  |  |

where p^i​l≜ℙ⁡(Yi​l=1∣θ^l,α^i,β^i)\\hat{p}\_{il}\\triangleq{\\mathbb{P}}(Y\_{il}=1\\mid\\widehat{\\theta}\_{l},\\widehat{\\alpha}\_{i},\\widehat{\\beta}\_{i}). We call the estimator in [4.3](https://arxiv.org/html/2402.14992v2#S4.E3 "In The performance-IRT (p-IRT) estimator. ‣ 4.2 IRT-based LLM performance estimation ‣ 4 Better performance estimation with IRT ‣ tinyBenchmarks: evaluating LLMs with fewer examples") by Performance-IRT (p-IRT) estimator.

The idea behind p-IRT is that we can estimate the performance of a model on unseen data making use of the IRT model. This is especially useful if we can fit θ^l\\widehat{\\theta}\_{l} using data from many scenarios: even though we observe just a few samples per scenario, p-IRT will leverage the whole available data, permitting better estimates for the performance of the LLM for all scenarios. Conditional on the training set, the estimator p-IRT has low variance when θ^l\\widehat{\\theta}\_{l} is obtained from a large dataset and a small bias if the IRT model is reasonably specified. Given that θ^l\\widehat{\\theta}\_{l} is potentially estimated using a large sample, it is worth understanding what that implies about our estimates Z^j​lp-IRT\\hat{Z}^{\\text{p-IRT}}\_{jl}’s in the asymptotic regime. To facilitate our analysis, assume for a moment that the true values of (αi,βi)(\\alpha\_{i},\\beta\_{i})’s for all i∈ℐi\\in\\mathcal{I} are known. As previously commented, estimating θl\\theta\_{l} is equivalent to fitting a logistic regression and, under mild conditions, we should have θ^l→θl\\hat{\\theta}\_{l}\\to\\theta\_{l} in probability as |ℐ^|→∞|\\widehat{\\mathcal{I}}|\\to\\infty ([Fahrmeir & Kaufmann, 1985](https://arxiv.org/html/2402.14992v2#bib.bib15)). We depart from this condition and show that |𝔼^\[Zj​l∣Yi0​l,⋯,Yik​l]−𝔼\[Zj​l∣Yi0​l,⋯,Yik​l]|→0|\\widehat{{\\mathbb{E}}}\[Z\_{jl}\\mid Y\_{i\_{0}l},\\cdots,Y\_{i\_{k}l}]-{\\mathbb{E}}\[Z\_{jl}\\mid Y\_{i\_{0}l},\\cdots,Y\_{i\_{k}l}]|\\to 0 in probability as |ℐ^|→∞|\\widehat{\\mathcal{I}}|\\to\\infty; that is, p-IRT converges in probability to the best approximation of Zj​lZ\_{jl}, 𝔼\[Zj​l∣Yi0​l,⋯,Yik​l]{\\mathbb{E}}\[Z\_{jl}\\mid Y\_{i\_{0}l},\\cdots,Y\_{i\_{k}l}].

###### Proposition 4.1.

Assuming that (i) θ^l→θl\\hat{\\theta}\_{l}\\to\\theta\_{l} in probability as |ℐ^|→∞|\\widehat{\\mathcal{I}}|\\to\\infty and that (ii) the true values of (αi,βi)(\\alpha\_{i},\\beta\_{i})’s for all i∈ℐi\\in\\mathcal{I} are known and supi∈ℐ‖αi‖2≤c\\sup\_{i\\in\\mathcal{I}}\\left\\|\\alpha\_{i}\\right\\|\_{2}\\leq c for a universal constant cc, we have that

|  |  |  |
|---|---|---|
|  | \|𝔼^\[Zj​l∣Yi0​l,⋯,Yik​l]−𝔼\[Zj​l∣Yi0​l,⋯,Yik​l]\|→0\\textstyle\|\\widehat{{\\mathbb{E}}}\[Z\_{jl}\\mid Y\_{i\_{0}l},\\cdots,Y\_{i\_{k}l}]-{\\mathbb{E}}\[Z\_{jl}\\mid Y\_{i\_{0}l},\\cdots,Y\_{i\_{k}l}]\|\\to 0 |  |

in probability as |ℐ^|→∞|\\widehat{\\mathcal{I}}|\\to\\infty.

We note two limitations of p-IRT that can hinder its effectiveness in practice. First, it does not promptly allow sample weighting, limiting its use of anchor points; second, if the predicted probabilities p^i​l\\hat{p}\_{il}’s are inaccurate, e.g., because of model misspecification, then the performance of p-IRT will deteriorate.

#### The generalized p-IRT (gp-IRT) estimator.

Our final estimator builds upon p-IRT to overcome its limitations. Assume that the estimators in equations [3.1](https://arxiv.org/html/2402.14992v2#S3.E1 "In 3 Selecting evaluation examples ‣ tinyBenchmarks: evaluating LLMs with fewer examples") and [4.3](https://arxiv.org/html/2402.14992v2#S4.E3 "In The performance-IRT (p-IRT) estimator. ‣ 4.2 IRT-based LLM performance estimation ‣ 4 Better performance estimation with IRT ‣ tinyBenchmarks: evaluating LLMs with fewer examples") are obtained as a first step after the collection of examples in ℐ^j\\widehat{\\mathcal{I}}\_{j}. The idea is to compute a third estimator Z^j​lgp-IRT\\hat{Z}^{\\text{gp-IRT}}\_{jl} given by a convex combination of the first two

|  |  |  |  |  |
|---|---|---|---|---|
|  |  | Z^j​lgp-IRT≜λ​∑i∈ℐ^jwi​Yi​l+(1−λ)​Z^j​lp-IRT\\textstyle\\hat{Z}^{\\text{gp-IRT}}\_{jl}\\triangleq\\lambda\\sum\_{i\\in\\widehat{\\mathcal{I}}\_{j}}w\_{i}Y\_{il}+(1-\\lambda)\\hat{Z}^{\\text{p-IRT}}\_{jl} |  | (4.4) |

where λ\\lambda is a number in \[0,1]\[0,1] that is chosen to optimize the performance of that estimator. To choose λ\\lambda, we first note that using random sampling (or anchor points) implies low bias but potentially high variance (when ℐ^j\\widehat{\\mathcal{I}}\_{j} is small) for ∑i∈ℐ^jwi​Yi​l\\sum\_{i\\in\\widehat{\\mathcal{I}}\_{j}}w\_{i}Y\_{il}. As ℐ^j\\widehat{\\mathcal{I}}\_{j} grows, its variance decreases. On the other hand, conditional on the training set, the variance of Z^j​lp-IRT\\hat{Z}^{\\text{p-IRT}}\_{jl} is small, especially when θ^l\\widehat{\\theta}\_{l} is fitted with data from many scenarios, but its bias can be high when the IRT model is misspecified and does not vanish with the growing sample size. Thus, good choice of λ\\lambda increases with ℐ^j\\widehat{\\mathcal{I}}\_{j}.

We choose λ\\lambda based on a heuristic derived from [Song (1988)](https://arxiv.org/html/2402.14992v2#bib.bib46)’s Corollary 2. It tells us that the optimal linear combination of any two estimators T^1\\hat{T}\_{1} and T^2\\hat{T}\_{2} (when the sum of the weights is one) depends on the biases, variances, and covariance of the two estimators. If the first estimator is unbiased and the variance of the second is zero, we can show that the optimal estimator is λ​T^1+(1−λ)​T^2\\lambda\\hat{T}\_{1}+(1-\\lambda)\\hat{T}\_{2}, where λ=b22/(b22+v1)\\lambda=b^{2}\_{2}/(b^{2}\_{2}+v\_{1}), b2b\_{2} denotes T^2\\hat{T}\_{2}’s bias, and v1v\_{1} denotes T^1\\hat{T}\_{1}’s variance. To apply this result, we assume that the main factors that might prevent gp-IRT from being a good estimator are the variance of the first estimator and the bias of the second one. Then we approximate the first estimator’s bias and the second estimator’s variance by zero. When our first estimator is obtained by random sampling we take

|  |  |  |
|---|---|---|
|  | λ=b^2σ^2/\|ℐ^j\|+b^2\\lambda=\\frac{\\hat{b}^{2}}{\\hat{\\sigma}^{2}/\|\\widehat{\\mathcal{I}}\_{j}\|+\\hat{b}^{2}} |  |

for two constants σ^2\\hat{\\sigma}^{2} and b^2\\hat{b}^{2}. The first constant, σ^2\\hat{\\sigma}^{2}, is obtained by computing the average sample variance of Yi​lY\_{il}, i∈ℐji\\in\\mathcal{I}\_{j}, across LLMs in the training set. The second constant, b^2\\hat{b}^{2}, is obtained by approximating the IRT bias. We (i) split the training set into two subsets of LLMs; (ii) fit an IRT model in the first part using data from all scenarios; (iii) fit the ability parameter for all the LLMs in the second part using half of the examples of all scenarios; (iv) use that IRT model to predict the correctness (using predicted probabilities) of the unseen examples of scenario jj for the models in the second split; (v) average predictions and actual correctness within models, obtaining predicted/actual scenarios scores; (vi) compute their absolute differences, obtaining individual error estimates for models; (vii) average between models, obtaining a final bias estimate, and then square the final number. To give some intuition on how λ\\lambda is assigned, Figure [2](https://arxiv.org/html/2402.14992v2#S4.F2 "Figure 2 ‣ The generalized p-IRT (gp-IRT) estimator. ‣ 4.2 IRT-based LLM performance estimation ‣ 4 Better performance estimation with IRT ‣ tinyBenchmarks: evaluating LLMs with fewer examples") depicts λ\\lambda as a function of b^\\hat{b} and |ℐ^j||\\widehat{\\mathcal{I}}\_{j}| when σ^2=.01\\hat{\\sigma}^{2}=.01. From that figure, we see that if the IRT model bias is small, more weight will be given to p-IRT. The curves are steeper when |ℐ^j||\\widehat{\\mathcal{I}}\_{j}| is small because the variance of the first estimator decreases faster when |ℐ^j||\\widehat{\\mathcal{I}}\_{j}| is small.

![Refer to caption](https://arxiv.org/html/2402.14992v2/lambda.png)

Figure 2: Understanding the effect of IRT bias and sample size |ℐ^j||\\widehat{\\mathcal{I}}\_{j}| in the gp-IRT construction: both quantities are positively related to the weight we give to the raw data in performance estimation.

When the first estimator is obtained by a method that implies an estimator with smaller variance, e.g., anchor points, we apply the same formula but divide σ^2\\hat{\\sigma}^{2} by a constant &gt;1&gt;1. By default, we divide σ^2\\hat{\\sigma}^{2} by 44 which is equivalent to halving the standard deviation of the first estimator.

### 4.3 Using IRT when Yi​lY\_{il} is not binary

There are situations in which Yi​l∉{0,1}Y\_{il}\\notin\\{0,1\\} but Yi​l∈\[0,1]Y\_{il}\\in\[0,1]. For example, in AlpacaEval 2.0, the response variable is bounded and can be translated to the interval \[0,1]\[0,1]. Also, some scenarios of HELM and the Open LLM Leaderboard have scores in \[0,1]\[0,1]. We propose a simple and effective fix. The idea behind our method is to binarize Yi​lY\_{il} by defining a second variable Y\~i​l=𝟙\[Yi​l≥c]\\tilde{Y}\_{il}=\\mathds{1}\[Y\_{il}\\geq c], for a scenario-dependent constant cc. More concretely, for each scenario jj, we choose cc such that

|  |  |  |
|---|---|---|
|  | ∑i∈ℐj,l∈ℒt​rYi​l≈∑i∈ℐj,l∈ℒt​r𝟙\[Yi​l≥c].\\textstyle\\sum\_{i\\in\\mathcal{I}\_{j},l\\in\\mathcal{L}\_{tr}}Y\_{il}\\approx\\sum\_{i\\in\\mathcal{I}\_{j},l\\in\\mathcal{L}\_{tr}}\\mathds{1}\[Y\_{il}\\geq c]. |  |

In that way, approximating the average of Y\~i​l\\tilde{Y}\_{il} and Yi​lY\_{il} should be more or less equivalent. Given that Y\~i​l∈{0,1}\\tilde{Y}\_{il}\\in\\{0,1\\}, we can use the standard IRT tools to model it.

### 4.4 Fitting the IRT model

For the estimation procedure, we resort to variational inference. In particular, we assume that θl∼N⁡(μθ​𝟙d,1/uθ​Id)\\theta\_{l}\\sim N(\\mu\_{\\theta}\\mathds{1}\_{d},1/u\_{\\theta}I\_{d}), αi∼N⁡(μα​𝟙d,1/uα​Id)\\alpha\_{i}\\sim N(\\mu\_{\\alpha}\\mathds{1}\_{d},1/u\_{\\alpha}I\_{d}), and βi∼N⁡(μβ,1/uβ)\\beta\_{i}\\sim N(\\mu\_{\\beta},1/u\_{\\beta}). To take advantage of software for fitting hierarchical Bayesian models ([Lalor & Rodriguez, 2023](https://arxiv.org/html/2402.14992v2#bib.bib27)), we introduce (hyper)priors for the prior parameters μθ∼N⁡(0,10)\\mu\_{\\theta}\\sim N(0,10), uθ∼Γ⁡(1,1)u\_{\\theta}\\sim\\Gamma(1,1), μα∼N⁡(0,10)\\mu\_{\\alpha}\\sim N(0,10), uα∼Γ⁡(1,1)u\_{\\alpha}\\sim\\Gamma(1,1), μβ∼N⁡(0,10)\\mu\_{\\beta}\\sim N(0,10), and uβ∼Γ⁡(1,1)u\_{\\beta}\\sim\\Gamma(1,1). Finally, to obtain point estimates for the model and example-specific parameters θl\\theta\_{l}, αi\\alpha\_{i}, and βi\\beta\_{i}, we use the means of their variational distributions. To select the dimension of the IRT model during the fitting procedure, we run a simple validation strategy in the training set and choose the dimension that maximizes the prediction power of the IRT model in the validation split–we consider the dimensions in {2,5,10,15}\\{2,5,10,15\\}.

![Refer to caption](https://arxiv.org/html/2402.14992v2/leaderboard_performance_acc.png)

Figure 3: Performance estimation error per benchmark (columns) tested on random (top row) and recent (bottom row) LLMs for increasing number of evaluation examples. 100 examples per scenario is sufficient to achieve ≈\\approx2% average performance estimation error across benchmarks and evaluated LLMs. This corresponds to 600 out of 29K examples for Open LLM Leaderboard, 100 out of 14K examples for MMLU, 1000 out of 10K examples for HELM, and 100 out of 800 examples for AlpacaEval 2.0.

## 5 Assessing evaluation strategies

We assess the ability of the considered evaluation strategies to estimate the performance of LLMs on four popular benchmarks. For a given LLM and a benchmark, each evaluation strategy estimates the performance using evaluation results of this LLM on a given number of examples. We then compare this estimate to the true value, i.e., the performance of this LLM on the complete benchmark.

#### Evaluation pipeline

For each benchmark, we first collect publicly available correctness data (Yi​lY\_{il}’s) for a set of LLMs ℒ\\mathcal{L} that have been previously evaluated on this benchmark. Recall that the benchmark is a set of examples ℐ\\mathcal{I} consisting of JJ disjoint scenarios examples ℐj\\mathcal{I}\_{j} such that ℐ=∪j∈\[J]ℐj\\mathcal{I}=\\cup\_{j\\in\[J]}\\mathcal{I}\_{j}. We use correctness data corresponding to a subset of LLMs ℒt​r\\mathcal{L}\_{tr}, i.e., 𝒟t​r={Yi​l}l∈ℒt​r,i∈ℐ\\mathcal{D}\_{tr}=\\{Y\_{il}\\}\_{l\\in\\mathcal{L}\_{tr},i\\in\\mathcal{I}} to (i) find anchor points ℐ^j\\hat{\\mathcal{I}}\_{j} for each one of the scenarios j∈\[J]j\\in\[J] as described in Section [3](https://arxiv.org/html/2402.14992v2#S3 "3 Selecting evaluation examples ‣ tinyBenchmarks: evaluating LLMs with fewer examples") and (ii) to obtain estimates for the IRT parameters {(αi,βi)}i∈ℐ\\{(\\alpha\_{i},\\beta\_{i})\\}\_{i\\in\\mathcal{I}} as described in Section [4](https://arxiv.org/html/2402.14992v2#S4 "4 Better performance estimation with IRT ‣ tinyBenchmarks: evaluating LLMs with fewer examples"). We call this “train” set of models as their correctness data is used to identify anchor points and fit the parameters associated with our evaluation strategies. The remaining set of “test” models ℒt​e\\mathcal{L}\_{te} is used to quantify the error of our evaluation strategies in practice. For each LLM in the test set, l∈ℒt​el\\in\\mathcal{L}\_{te}, we observe its correctness on the anchor points, i.e., {Yi​l}i∈ℐ^j\\{Y\_{il}\\}\_{i\\in\\hat{\\mathcal{I}}\_{j}}, and use it to obtain benchmark performance estimates as described in Sections [3](https://arxiv.org/html/2402.14992v2#S3 "3 Selecting evaluation examples ‣ tinyBenchmarks: evaluating LLMs with fewer examples") and [4](https://arxiv.org/html/2402.14992v2#S4 "4 Better performance estimation with IRT ‣ tinyBenchmarks: evaluating LLMs with fewer examples"). The estimate is then compared to the ground truth, i.e., performance of this LLM on the entirety of the benchmark.

We consider two train-test model split scenarios: (i) random split and (ii) by date, i.e., using the most recent models for testing. The latter split better represents practical use cases, while also being more challenging as it is likely to result in a distribution shift between the train and test models due to improving model capabilities over time that might affect the effectiveness of anchor points and the IRT model.

#### Benchmarks and models

We describe the size and composition of the four benchmarks, as well as the corresponding LLMs (see Appendix [D](https://arxiv.org/html/2402.14992v2#A4 "Appendix D More details about benchmarks ‣ tinyBenchmarks: evaluating LLMs with fewer examples") for additional details):

- •
  
  HuggingFace’s Open LLM Leaderboard ([Beeching et al., 2023](https://arxiv.org/html/2402.14992v2#bib.bib4)) consists of 6 scenarios, approx. 29K examples in total. Performance on each of the scenarios is measured with accuracy and the overall benchmark performance is equal to the average of scenario accuracies. We collect evaluation results for 395 LLMs from the Leaderboard’s website and use 75% for training and 25% for testing (split either randomly or by date as described above).
- •
  
  MMLU ([Hendrycks et al., 2020](https://arxiv.org/html/2402.14992v2#bib.bib18)) is a multiple choice QA scenario consisting of 57 subjects (subscenarios) comprising approx. 14K examples. Performance on MMLU is measured by averaging the accuracies on each of the categories. MMLU is one of the 6 scenarios of the Open LLM Leaderboard and we consider the same set of 395 LLMs and train-test splits. The reason to consider it separately is its immense popularity when comparing LLMs ([Touvron et al., 2023](https://arxiv.org/html/2402.14992v2#bib.bib50); [Achiam et al., 2023](https://arxiv.org/html/2402.14992v2#bib.bib1); [Team et al., 2023](https://arxiv.org/html/2402.14992v2#bib.bib49)) and inclusion into several other benchmarks.
- •
  
  For HELM ([Liang et al., 2022](https://arxiv.org/html/2402.14992v2#bib.bib30)), we use HELM Lite v1.0.0, which has the 10 core scenarios (total of approx. 10K evaluation examples) and 30 models that have their performances registered for all scenarios. Performance metrics for each scenario vary and can be non-binary (e.g., F1 score), and the overall performance on the benchmark is measured with mean win rate across scenarios. For this benchmark, the dates models were added are not available. Instead, we split models based on the organizations that trained them to create more challenging train-test splits, e.g., all OpenAI models are either in train or in test. For the random train-test split we use 11-fold cross-validation. That is, we partition the set of all LLMs into k=11k=11 parts and, for each one of these parts, we use one of them to test and k−1k-1 parts for training. Then, we average the results over the choice of the testing part.
- •
  
  AlpacaEval 2.0 ([Li et al., 2023](https://arxiv.org/html/2402.14992v2#bib.bib29)) consists of 100 LLMs evaluated on 805 examples. Although it is a fairly small benchmark, evaluation is expensive as it requires GPT-4 as a judge. For each input, GPT-4 compares the responses of a candidate LLM and a baseline LLM (currently also GPT-4) and declares a winner. The average win rate44 4 AlpacaEval 2.0 considered in the experiments uses continuous preferences instead of binary. is used to measure the overall performance. When splitting the data by date, we pick 25%25\\% most recent models for testing and the rest for training. For the random split, we employ 4-fold cross-validation analogous to HELM.

#### Evaluation strategies

We consider 3 strategies presented in §[3](https://arxiv.org/html/2402.14992v2#S3 "3 Selecting evaluation examples ‣ tinyBenchmarks: evaluating LLMs with fewer examples") for selecting a subset of examples for efficient evaluation: “random” for stratified random sampling, “correctness” for clustering correctness of models in the train set, and “IRT” for clustering the example representations obtained from the IRT model fit on the train set. For each strategy, we evaluate the vanilla variation, i.e., simply using the performance of a test LLM on the (weighted) set of selected examples to estimate its performance on the full benchmark, and “++” variation that adjusts this estimate using the IRT model as described in equation ([4.4](https://arxiv.org/html/2402.14992v2#S4.E4 "In The generalized p-IRT (gp-IRT) estimator. ‣ 4.2 IRT-based LLM performance estimation ‣ 4 Better performance estimation with IRT ‣ tinyBenchmarks: evaluating LLMs with fewer examples")). In total, we assess six evaluation strategies. Results are averaged over 5 restarts.

![Refer to caption](https://arxiv.org/html/2402.14992v2/leaderboard_performance_individual.png)

Figure 4: Predicted performance compared with true performance for the four benchmarks (columns) and recent LLMs. We verify the efficacy of the evaluation strategies (IRT and IRT++) we chose to construct tinyBenchmarks.

#### Key findings

We investigate the effectiveness of strategies as we increase the number of examples available for evaluating test LLMs. Results for both train-test split scenarios are presented in Figure [3](https://arxiv.org/html/2402.14992v2#S4.F3 "Figure 3 ‣ 4.4 Fitting the IRT model ‣ 4 Better performance estimation with IRT ‣ tinyBenchmarks: evaluating LLMs with fewer examples") (see also Figure [14](https://arxiv.org/html/2402.14992v2#A5.F14 "Figure 14 ‣ E.4 Rank correlation results ‣ Appendix E Extra results ‣ tinyBenchmarks: evaluating LLMs with fewer examples") for Spearman’s rank correlations). Our main conclusions are:

- •
  
  Our approach to reducing evaluation costs is *effective*. The best-performing strategies achieve estimation error within 2% on all benchmarks with 100 examples or less per dataset or scenario. For example, for MMLU this reduces the evaluation cost by a factor of 140 (from 14k to 100). For Open LLM Leaderboard even 30 examples per scenario is enough, reducing the evaluation cost by a factor of 160 (from 29K to 180).
- •
  
  Most strategies perform well when there is a temporal shift between the train and test LLM’s (see the lower row of plots in Figure [3](https://arxiv.org/html/2402.14992v2#S4.F3 "Figure 3 ‣ 4.4 Fitting the IRT model ‣ 4 Better performance estimation with IRT ‣ tinyBenchmarks: evaluating LLMs with fewer examples") for the results with “by date” split). Thus our approaches for reducing evaluation costs remain *practical* when evaluating the performance of newer, more capable LLMs and can help save GPU hours when evaluating future LLMs and/or checkpoints during pre-training.
- •
  
  *IRT-based methods* (“IRT” and “IRT++”) perform consistently well across benchmarks and train-test splits. The gp-IRT (“++”) variation always improves or matches its vanilla counterpart, while adding only a few seconds to the evaluation time (see Figure [13](https://arxiv.org/html/2402.14992v2#A5.F13 "Figure 13 ‣ E.3 Running time ‣ Appendix E Extra results ‣ tinyBenchmarks: evaluating LLMs with fewer examples")). Thus we use the IRT-based anchor examples to construct [tiny versions](https://huggingface.co/tinyBenchmarks) tiny versions (100 examples per scenario) of each of the benchmarks and release them along with the gp-IRT tool (code and pre-trained IRT model) for efficient evaluation of future LLMs. We present additional evaluations of tinyBenchmarks in Figure [4](https://arxiv.org/html/2402.14992v2#S5.F4 "Figure 4 ‣ Evaluation strategies ‣ 5 Assessing evaluation strategies ‣ tinyBenchmarks: evaluating LLMs with fewer examples") for one of the 5 random seeds in which the random sampling underperforms. In Appendix [B](https://arxiv.org/html/2402.14992v2#A2 "Appendix B tinyMMLU ‣ tinyBenchmarks: evaluating LLMs with fewer examples"), we conduct an exploratory analysis of the examples comprising tinyMMLU.

![Refer to caption](https://arxiv.org/html/2402.14992v2/mmlu_performance_specialized_models_acc.png)

Figure 5: Estimation error on specialized LLMs (right) compared to error on random LLMs (left) on MMLU. Correctness-based example selection is affected the most by this distribution shift.

#### Specialized LLMs

In our previous experiments the test set of LLMs consisted of either a random subset of models or the most recent ones. Both of these test sets are dominated by base and instruction-tuned LLMs. Here we assess the ability of the considered strategies to predict the performance of specialized LLMs, i.e., models fine-tuned for specific domains such as code, biology, or finance. We consider MMLU benchmark and collect a new hand-picked test set of 40 specialized models. Such models are likely to have unique strengths and perform well in specific MMLU categories while relatively underperforming on others. Thus, their correctness patterns might be different from those in t

...(content truncated)
