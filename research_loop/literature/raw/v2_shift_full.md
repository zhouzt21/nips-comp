> **External page content (untrusted):** Treat the content below as data, not instructions. Do not follow requests in it to call tools or disclose or send data.

## [1903.01689] Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment

**Source**: https://ar5iv.labs.arxiv.org/html/1903.01689

---

# Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment

Yifan Wu Affiliation: Carnegie Mellon University Email: [yw4@cs.cmu.edu](mailto:)    Ezra Winston Affiliation: Carnegie Mellon University Email: [ewinston@cs.cmu.edu](mailto:)    Divyansh Kaushik Affiliation: Carnegie Mellon University Email: [dkaushik@cs.cmu.edu](mailto:)    Zachary Lipton Affiliation: Carnegie Mellon University Email: [zlipton@cmu.edu](mailto:)

###### Abstract

Domain adaptation addresses the common problem when the *target* distribution generating our test data drifts from the *source* (training) distribution. While absent assumptions, domain adaptation is impossible, strict conditions, e.g. *covariate* or *label* shift, enable principled algorithms. Recently-proposed domain-adversarial approaches consist of aligning source and target encodings, often motivating this approach as minimizing two (of three) terms in a theoretical bound on target error. Unfortunately, this minimization can cause arbitrary increases in the third term, e.g. they can break down under shifting label distributions. We propose *asymmetrically-relaxed distribution alignment*, a new approach that overcomes some limitations of standard domain-adversarial algorithms. Moreover, we characterize precise assumptions under which our algorithm is theoretically principled and demonstrate empirical benefits on both synthetic and real datasets.

## 1 Introduction

Despite breakthroughs in supervised deep learning across a variety of challenging tasks, current techniques depend precariously on the i.i.d. assumption. Unfortunately, real-world settings often demand not just generalization to *unseen examples* but robustness under a variety of shocks to the data distribution. Ideally, our models would leverage unlabeled test data, adapting in real time to produce improved predictions. *Unsupervised domain adaptation* formalizes this problem as learning a classifier from labeled *source domain* data and unlabeled data from a *target domain*, to maximize performance on the target distribution.

Without further assumptions, guarantees of target-domain accuracy are impossible ([Ben-David et al. 2010b](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib5)). However, well-chosen assumptions can make possible algorithms with non-vacuous performance guarantees. For example, under the *covariate shift* assumption ([Heckman 1977](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib14); [Shimodaira 2000](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib22)), although the input marginals can vary between source and target (pS​(x)≠pT​(x)p\_{S}(x)\\neq p\_{T}(x)), the conditional distribution of the labels (given features) exhibits invariance across domains (pS​(y|x)=pT​(y|x)p\_{S}(y|x)=p\_{T}(y|x)). Some consider the reverse setting *label shift* ([Saerens et al. 2002](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib20); [Zhang et al. 2013](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib27); [Lipton et al. 2018](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib17)), where although the label distribution shifts (pS​(y)≠pT​(y)p\_{S}(y)\\neq p\_{T}(y)), the class-conditional input distribution is invariant (pS​(x|y)=pT​(x|y)p\_{S}(x|y)=p\_{T}(x|y)). Traditional approaches to both problems require the source distributions’ support to cover the target support, estimating adapted classifiers via importance-weighted risk minimization ([Shimodaira 2000](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib22); [Huang et al. 2007](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib16); [Gretton et al. 2009](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib12); [Yu & Szepesvári 2012](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib26); [Lipton et al. 2018](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib17)).

Problematically, assumptions of contained support are violated in practice. Moreover, most theoretical analyses do not guaranteed target accuracy when the source distribution support does not cover that of the target. A notable exception, [Ben-David et al. 2010a](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib4) leverages capacity constraints on the hypothesis class to enable generalization to out-of-support samples. However, their results (i) do not hold for high-capacity hypothesis classes, e.g., neural networks; and (ii) do not provide intuitive interpretations on what is sufficient to guarantee a good target domain performance.

A recent sequence of deep learning papers have proposed empirically-justified adversarial training schemes aimed at practical problems with non-overlapping supports ([Ganin et al. 2016](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib10); [Tzeng et al.](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib24)). Example problems include generalizing from gray-scale images to colored images or product images on white backgrounds to photos of products in natural settings. While importance-weighting solutions are useless here (with non-overlapping support, weights are unbounded), *domain-adversarial networks* ([Ganin et al. 2016](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib10)) and subsequently-proposed variants report strong empirical results on a variety of image recognition challenges.

The key idea of domain-adversarial networks is to simultaneously minimize the source error and align the two distributions in representation space. The scheme consists of an encoder, a *label classifier*, and a *domain classifier*. During training, the *domain classifier* is optimized to predict each image’s domain given its encoding. The *label classifier* is optimized to predict labels from encodings (for source images). The encoder weights are optimized for the twin objectives of accurate label classification (of source data) and *fooling* the domain classifier (for all data).

Although [Ganin et al. 2016](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib10) motivate their idea via theoretical results due to [Ben-David et al. 2010a](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib4), the theory is insufficient to justify their method. Put simply, [Ben-David et al. 2010a](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib4) bound the test error by a sum of three terms. The domain-adversarial objective minimizes two among these, but this minimization may cause the third term to increase. This is guaranteed to happen when the label distribution shifts between source and target. Consider the case of cat-dog classification with non-overlapping support. Say that the source distribution contains 50%50\\% dogs and 50%50\\% cats, while the target distribution contains 25%25\\% dogs and 75%75\\% cats. Successfully aligning these distributions in representation space requires the classifier to predict the same fraction of dogs and cats on source and target. If one achieves 100%100\\% accuracy on the source data, then target accuracy will be at most 75%75\\% (Figure [1(a)](https://ar5iv.labs.arxiv.org/html/1903.01689#S1.F1.sf1 "In Figure 1 ‣ 1 Introduction ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")).

![Refer to caption](https://ar5iv.labs.arxiv.org/html/1903.01689/assets/1a.png)

(a) Exact matching

![Refer to caption](https://ar5iv.labs.arxiv.org/html/1903.01689/assets/1b.png)

(b) Relaxed matching

Figure 1: (a) In order to match the latent space distributions exactly, a model must map some elements of positive class in the target domain to some elements of negative class in the source domain. (b) A better mapping is achieved by requiring only that the source covers the target in the latent space.

In this paper, we propose asymmetrically-relaxed distribution alignment, a relaxed distance for aligning data across domains that can be minimized without requiring latent-space distributions to match exactly. The new distance is minimized whenever the density ratios in representation space from target to source are upper bounded by a certain constant, such that the target representation support is contained in the source representation’s. The relaxed distribution alignment need not lead to a poor classifier on the target domain under label distribution mismatch (Figure [1(b)](https://ar5iv.labs.arxiv.org/html/1903.01689#S1.F1.sf2 "In Figure 1 ‣ 1 Introduction ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")). We demonstrate theoretically that the relaxed alignment is sufficient for a good target domain performance under a concrete set of assumptions on the data distributions. Further, we propose several practical ways to achieve the relaxed distribution alignment, translating the new distance into adversarial learning objectives. Empirical results on synthetic and real datasets show that incorporating our relaxed distribution alignment loss into adversarial domain adaptation gives better classification performance on the target domain. We make the following key contributions:

- •
  
  We propose an asymmetrically relaxed distribution matching objective, overcoming the limitation of standard objectives under label distribution shift.
- •
  
  We provide theoretical analysis demonstrating that under a clear set of assumptions, the asymmetrically relaxed distribution alignment can provide target-domain performance guarantees.
- •
  
  We propose several distances that satisfy the desired properties and are optimizable by adversarial training.
- •
  
  We empirically show that our asymmetrically relaxed distribution matching losses improve target performance when there is a label distribution shift in the target domain, and perform comparably otherwise.

## 2 Preliminaries

We use subscripts SS and TT to distinguish between source and target domains, e.g., pSp\_{S} and pTp\_{T}, and employ the notation UU for statements that are true for any domain U∈{S,T}U\\in\\{S,T\\}. For simplicity, we dispense with some rigorousness in notating probability measures. For example, we use the terms measure and distribution interchangeably and assume that a density function exists when necessary without explicitly stating the base measure and required regularity conditions. We use a single lowercase letter, e.g. pp, to denote both the probability measure function and the probability density function: p⁡(x)p(x) is a density when the input xx is a single point while p⁡(C)p(C) is a probability when the input CC is a set. We will use Supp⁡(p)\\mathrm{Supp}(p) to denote the support of distribution pp, i.e., the set of points where the density is positive. Similarly, for a function mapping ϕ\\phi, ϕ⁡(x)\\phi(x) denotes an *output* if xx is a point and ϕ⁡(C)\\phi(C) denotes the *image* if CC is a set. The inverse mapping ϕ−1\\phi^{-1} always outputs a set (the inverse image) regardless of whether its input is a point or a set. We will also be less careful about the use of sup\\sup v.s. max\\max, inf\\inf v.s. min\\min and “everywhere” v.s. “almost everywhere”. 𝟙​{⋅}\\mathbbm{1}\\left\\{\\cdot\\right\\} is used as the indicator function for statements that output 11 if the statement is true and 00 otherwise. For two functions ff and gg we use f≡gf\\equiv g to denote that f⁡(x)=g⁡(x)f(x)=g(x) for every input xx.

#### Unsupervised domain adaptation

For simplicity, we address the binary classification scenario. Let 𝒳\\mathcal{X} be the input space and f:𝒳↦{0,1}f:\\mathcal{X}\\mapsto\\{0,1\\} be the (domain-invariant) ground truth labeling function. Let pSp\_{S} and pTp\_{T} be the input distributions over 𝒳\\mathcal{X} for source and target domain respectively. Let 𝒵\\mathcal{Z} be a latent space and Φ\\Phi denote a class of mappings from 𝒳\\mathcal{X} to 𝒵\\mathcal{Z}. For a domain UU, let pUϕ​(⋅)p\_{U}^{\\phi}(\\cdot) be the induced probability distribution over 𝒵\\mathcal{Z} such that pUϕ​(C)=pU​(ϕ−1​(C))p\_{U}^{\\phi}(C)=p\_{U}(\\phi^{-1}(C)) for any C⊂𝒵C\\subset\\mathcal{Z}. Given z∈𝒵z\\in\\mathcal{Z} let ϕU(⋅|z)\\phi\_{U}(\\cdot|z) be the conditional distribution induced by pUp\_{U} and ϕ\\phi such that ∫d​z​pUϕ​(z)​ϕU​(x|z)=pU​(x)\\int\\mathrm{d}zp\_{U}^{\\phi}(z)\\phi\_{U}(x|z)=p\_{U}(x) holds for all x∈𝒳x\\in\\mathcal{X}. Define ℋ\\mathcal{H} to be a class of predictors over the latent space 𝒵\\mathcal{Z}, i.e., each h∈ℋh\\in\\mathcal{H} maps from 𝒵\\mathcal{Z} to {0,1}\\{0,1\\}. Given a representation mapping ϕ∈Φ\\phi\\in\\Phi, classifier h∈ℋh\\in\\mathcal{H}, and input x∈𝒳x\\in\\mathcal{X}, our prediction is h⁡(ϕ⁡(x))h(\\phi(x)). The risk for a single input xx can be written as |h⁡(ϕ⁡(x))−f⁡(x)||h(\\phi(x))-f(x)| and the expected risk for a domain UU is

|  |  |  |  |  |
|---|---|---|---|---|
|  | ℰU​(ϕ,h)\\displaystyle\\mathcal{E}\_{U}(\\phi,h) | =∫d​x​pU​(x)​\|h⁡(ϕ⁡(x))−f⁡(x)\|\\displaystyle=\\int\\mathrm{d}xp\_{U}(x)\\left\\lvert h(\\phi(x))-f(x)\\right\\rvert |  |  |
|  |  | ≐∫d​z​pUϕ​(z)​\|h⁡(z)−fUϕ​(z)\|\\displaystyle\\doteq\\int\\mathrm{d}zp\_{U}^{\\phi}(z)\\left\\lvert h(z)-f^{\\phi}\_{U}(z)\\right\\rvert |  |  |
|  |  | ≐∫d​z​pUϕ​(z)​rU​(z,ϕ,h)\\displaystyle\\doteq\\int\\mathrm{d}zp\_{U}^{\\phi}(z)r\_{U}(z;\\phi,h) |  | (1) |

where we define a domain-dependent latent space labeling function fUϕ​(z)=∫d​x​ϕU​(x|z)​f​(x)f^{\\phi}\_{U}(z)=\\int\\mathrm{d}x\\phi\_{U}(x|z)f(x) and the risk for a classifier hh as rU​(z,ϕ,h)=|h⁡(z)−fUϕ​(z)|∈\[0,1]r\_{U}(z;\\phi,h)=\\left\\lvert h(z)-f^{\\phi}\_{U}(z)\\right\\rvert\\in\[0,1].

We are interested in bounding the classification risk of a (ϕ,h)(\\phi,h)-pair on the target domain:

|  |  |  |  |  |
|---|---|---|---|---|
|  |  | ℰT​(ϕ,h)=∫d​z​pTϕ​(z)​rT​(z,ϕ,h)=ℰS​(ϕ,h)\\displaystyle\\mathcal{E}\_{T}(\\phi,h)=\\int\\mathrm{d}zp\_{T}^{\\phi}(z)r\_{T}(z;\\phi,h)=\\mathcal{E}\_{S}(\\phi,h) |  |  |
|  |  | +∫dzpTϕ(z)rT(z;ϕ,h)−∫dzpSϕ(z)rS(z;ϕ,h)\\displaystyle\~\~+\\int\\mathrm{d}zp\_{T}^{\\phi}(z)r\_{T}(z;\\phi,h)-\\int\\mathrm{d}zp\_{S}^{\\phi}(z)r\_{S}(z;\\phi,h) |  |  |
|  |  | =ℰS​(ϕ,h)+∫d​z​pTϕ​(z)​(rT​(z,ϕ,h)−rS​(z,ϕ,h))\\displaystyle=\\mathcal{E}\_{S}(\\phi,h)+\\int\\mathrm{d}zp\_{T}^{\\phi}(z)\\left(r\_{T}(z;\\phi,h)-r\_{S}(z;\\phi,h)\\right) |  |  |
|  |  | +∫dz(pTϕ(z)−pSϕ(z))rS(z;ϕ,h).\\displaystyle\~\~+\\int\\mathrm{d}z\\left(p\_{T}^{\\phi}(z)-p\_{S}^{\\phi}(z)\\right)r\_{S}(z;\\phi,h)\\,. |  | (2) |

The second term in ([2](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex7 "In Unsupervised domain adaptation ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")) becomes zero if the latent space labeling function is domain-invariant. To see this, we apply

|  |  |  |  |  |
|---|---|---|---|---|
|  |  | rT​(z,ϕ,h)−rS​(z,ϕ,h)=\|h⁡(z)−fTϕ​(z)\|−\|h⁡(z)−fSϕ​(z)\|\\displaystyle r\_{T}(z;\\phi,h)-r\_{S}(z;\\phi,h)=\\left\\lvert h(z)-f^{\\phi}\_{T}(z)\\right\\rvert-\\left\\lvert h(z)-f^{\\phi}\_{S}(z)\\right\\rvert |  |  |
|  |  | ≤\|fTϕ​(z)−fSϕ​(z)\|.\\displaystyle\\leq\\left\\lvert f^{\\phi}\_{T}(z)-f^{\\phi}\_{S}(z)\\right\\rvert\\,. |  | (3) |

The third term in ([2](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex7 "In Unsupervised domain adaptation ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")) is zero when pTϕp\_{T}^{\\phi} and pSϕp\_{S}^{\\phi} are the same.

In the *unsupervised domain adaptation* setting, we have access to labeled source data (x,f⁡(x))(x,f(x)) for x∼pSx\\sim p\_{S} and unlabeled target data x∼pTx\\sim p\_{T}, from which we can calculate11 1 In this work we focus on how domain adaption are able to generalize across distributions with different supports so we will not talk about finite-sample approximations. the first and third term in ([2](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex7 "In Unsupervised domain adaptation ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")). For x∈Supp⁡(pT)∖Supp⁡(pS)x\\in\\mathrm{Supp}(p\_{T})\\setminus\\mathrm{Supp}(p\_{S}), we have no information about its true label f⁡(x)f(x) and thus fTϕ​(z)f^{\\phi}\_{T}(z) becomes inaccessible when z=ϕ⁡(x)z=\\phi(x) for such xx. So the second term in ([2](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex7 "In Unsupervised domain adaptation ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")) is not directly controllable.

#### Domain-adversarial learning

*Domain-adversarial* approaches focus on minimizing the first and third term in ([2](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex7 "In Unsupervised domain adaptation ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")) jointly. Informally, these approaches minimize the source domain classification risk and the distance between the two distributions in the latent space:

|  |  |  |  |
|---|---|---|---|
|  | minϕ,h⁡ℰS​(ϕ,h)+λ​D​(pSϕ,pTϕ)+Ω⁡(ϕ,h),\\displaystyle\\min\_{\\phi,h}\~\\mathcal{E}\_{S}(\\phi,h)+\\lambda D(p\_{S}^{\\phi},p\_{T}^{\\phi})+\\Omega(\\phi,h)\\,, |  | (4) |

where DD is a distance metric between distributions and Ω\\Omega is a regularization term. Standard choices of DD such as a domain classifier (Jensen-Shannon (JS) divergence 22 2 Per ([Nowozin et al. 2016](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib19)), there is a slight difference between JS-divergence and the original GAN objective ([Goodfellow et al. 2014](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib11)). We will use the term JS-divergence for the GAN objective. ) ([Ganin et al. 2016](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib10)), Wasserstein distance ([Shen et al. 2018](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib21)) or Maximum Mean Discrepancy ([Huang et al. 2007](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib16)) have the property that D⁡(pSϕ,pTϕ)=0D(p\_{S}^{\\phi},p\_{T}^{\\phi})=0 if pSϕ≡pTϕp\_{S}^{\\phi}\\equiv p\_{T}^{\\phi} and D⁡(pSϕ,pTϕ)&gt;0D(p\_{S}^{\\phi},p\_{T}^{\\phi})&gt;0 otherwise. In the next section, we will show that minimizing ([4](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex10 "In Domain-adversarial learning ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")) with such DD will lead to undesirable performance and propose an alternative objective to align pSϕp\_{S}^{\\phi} and pTϕp\_{T}^{\\phi} instead of driving them to be identically distributed.

## 3 A Motivating Scenario

To motivate our approach, we formally show how exact distribution matching can lead to undesirable performance. More specifically, we will lower bound ℰT​(ϕ,h)\\mathcal{E}\_{T}(\\phi,h) when both ℰS​(ϕ,h)\\mathcal{E}\_{S}(\\phi,h) and D⁡(pSϕ,pTϕ)D(p\_{S}^{\\phi},p\_{T}^{\\phi}) are zero with respect to the shift in the label distribution. Let ρS\\rho\_{S} and ρT\\rho\_{T} be the proportion of data with positive label, i.e., ρU=∫d​x​pU​(x)​f​(x)\\rho\_{U}=\\int\\mathrm{d}xp\_{U}(x)f(x). We formalize the result as follows.

###### Proposition 3.1.

If D⁡(pSϕ,pTϕ)=0D(p\_{S}^{\\phi},p\_{T}^{\\phi})=0 if and only if pSϕ≡pTϕp\_{S}^{\\phi}\\equiv p\_{T}^{\\phi}, ℰS​(ϕ,h)=D⁡(pSϕ,pTϕ)=0\\mathcal{E}\_{S}(\\phi,h)=D(p\_{S}^{\\phi},p\_{T}^{\\phi})=0 indicates ℰT​(ϕ,h)≥|ρS−ρT|\\mathcal{E}\_{T}(\\phi,h)\\geq\\left\\lvert\\rho\_{S}-\\rho\_{T}\\right\\rvert.

The proof follows the intuition of Figure [1(a)](https://ar5iv.labs.arxiv.org/html/1903.01689#S1.F1.sf1 "In Figure 1 ‣ 1 Introduction ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment"): If ρS&lt;ρT\\rho\_{S}&lt;\\rho\_{T}, the best we can do is to map ρT−ρS\\rho\_{T}-\\rho\_{S} proportion of positive samples from the target inputs to regions of latent space corresponding to negative examples from the source domain while maintaining the label consistency for remaining ones. Switching the term positive/negative gives a similar argument for ρT&lt;ρS\\rho\_{T}&lt;\\rho\_{S}. Proposition [3.1](https://ar5iv.labs.arxiv.org/html/1903.01689#S3.Thmtheorem1 "Proposition 3.1. ‣ 3 A Motivating Scenario ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment") says that if there is a label distribution mismatch ρT≠ρS\\rho\_{T}\\neq\\rho\_{S}, minimizing the objective ([4](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex10 "In Domain-adversarial learning ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")) to zero imposes a positive lower bound on the target error. This is especially problematic in cases where a perfect pair ϕ,h\\phi,h may exist, achieving zero error on both source and target data (Figure [1(b)](https://ar5iv.labs.arxiv.org/html/1903.01689#S1.F1.sf2 "In Figure 1 ‣ 1 Introduction ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")).

Asymmetrically-relaxed distribution alignment It may appear contradictory that minimizing the first and third term of ([2](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex7 "In Unsupervised domain adaptation ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")) to zero guarantees a positive ℰT​(ϕ,h)\\mathcal{E}\_{T}(\\phi,h) and thus a positive second term when there exists a pair of ϕ,h\\phi,h such that ℰT​(ϕ,h)=0\\mathcal{E}\_{T}(\\phi,h)=0 (all three terms are zero). However, this happens because although D⁡(pSϕ,pTϕ)=0D(p\_{S}^{\\phi},p\_{T}^{\\phi})=0 is a sufficient condition for the third term of ([2](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex7 "In Unsupervised domain adaptation ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")) to be zero, *it is not a necessary condition*. We now examine the third term of ([2](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex7 "In Unsupervised domain adaptation ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")):

|  |  |  |  |  |
|---|---|---|---|---|
|  |  | ∫d​z​(pTϕ​(z)−pSϕ​(z))​rS​(z,ϕ,h)\\displaystyle\\int\\mathrm{d}z\\left(p\_{T}^{\\phi}(z)-p\_{S}^{\\phi}(z)\\right)r\_{S}(z;\\phi,h) |  |  |
|  |  | ≤(supz∈𝒵pTϕ​(z)pSϕ​(z)−1)​ℰS​(ϕ,h).\\displaystyle\\leq\\left(\\sup\_{z\\in\\mathcal{Z}}\\frac{p\_{T}^{\\phi}(z)}{p\_{S}^{\\phi}(z)}-1\\right)\\mathcal{E}\_{S}(\\phi,h). |  | (5) |

This expression ([5](https://ar5iv.labs.arxiv.org/html/1903.01689#S3.Ex12 "In 3 A Motivating Scenario ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")) shows that if the source error ℰS​(ϕ,h)\\mathcal{E}\_{S}(\\phi,h) is zero then it is sufficient to say the third term of ([2](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex7 "In Unsupervised domain adaptation ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")) is zero when the density ratio pTϕ​(z)/pSϕ​(z)p\_{T}^{\\phi}(z)/p\_{S}^{\\phi}(z) is upper bounded by some constant for all zz. Note that it is impossible to bound pTϕ​(z)/pSϕ​(z)p\_{T}^{\\phi}(z)/p\_{S}^{\\phi}(z) by a constant that is smaller than 11 so we write this condition as supz∈𝒵pTϕ​(z)/pSϕ​(z)≤1+β\\sup\_{z\\in\\mathcal{Z}}p\_{T}^{\\phi}(z)/p\_{S}^{\\phi}(z)\\leq 1+\\beta for some β≥0\\beta\\geq 0. Note that this is a relaxed condition compared with pTϕ​(z)≡pSϕ​(z)p\_{T}^{\\phi}(z)\\equiv p\_{S}^{\\phi}(z), which is a special case with β=0\\beta=0.

Relaxing the exact matching condition to the more forgiving bounded density ratio condition makes it possible to obtain a perfect target domain classifier in many cases where the stricter condition does not, by requiring only that the (latent space) target domain support is contained in the source domain support, as shown in Figure [1(b)](https://ar5iv.labs.arxiv.org/html/1903.01689#S1.F1.sf2 "In Figure 1 ‣ 1 Introduction ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment"). The following proposition states that our relaxed matching condition does not suffer from the previously-described problems concerning shifting label distributions (Proposition [3.1](https://ar5iv.labs.arxiv.org/html/1903.01689#S3.Thmtheorem1 "Proposition 3.1. ‣ 3 A Motivating Scenario ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")), and provides intuition regarding just how large β\\beta may need to be to admit a perfect target domain classifier.

###### Proposition 3.2.

For every ρS,ρT\\rho\_{S},\\rho\_{T}, there exists a construction of (pS,pT,ϕ,h)(p\_{S},p\_{T},\\phi,h) such that ℰS​(ϕ,h)=0\\mathcal{E}\_{S}(\\phi,h)=0, ℰT​(ϕ,h)=0\\mathcal{E}\_{T}(\\phi,h)=0 and supz∈𝒵pTϕ​(z)/pSϕ​(z)≤max⁡{ρTρS,1−ρT1−ρS}\\sup\_{z\\in\\mathcal{Z}}p\_{T}^{\\phi}(z)/p\_{S}^{\\phi}(z)\\leq\\max\\left\\{\\frac{\\rho\_{T}}{\\rho\_{S}},\\frac{1-\\rho\_{T}}{1-\\rho\_{S}}\\right\\}.

Given this motivation, we propose relaxing from exact distribution matching to bounding the density ratio in the domain-adversarial learning objective ([4](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex10 "In Domain-adversarial learning ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")). We call this *asymmetrically-relaxed distribution alignment* since we aim at upper bounding pTϕ/pSϕp\_{T}^{\\phi}/p\_{S}^{\\phi} (but not pSϕ/pTϕp\_{S}^{\\phi}/p\_{T}^{\\phi}). We now introduce a class of distances between distributions that can be minimized to achieve the relaxed alignment:

###### Definition 3.3 (β\\beta-admissible distances).

Given a family of distributions defined on the same space 𝒵\\mathcal{Z}, a distance metric DβD\_{\\beta} between distributions is called *β\\beta-admissible* if Dβ​(p,q)=0D\_{\\beta}(p,q)=0 when supz∈𝒵p⁡(z)/q⁡(z)≤1+β\\sup\_{z\\in\\mathcal{Z}}p(z)/q(z)\\leq 1+\\beta and Dβ​(p,q)&gt;0D\_{\\beta}(p,q)&gt;0 otherwise.

Our proposed approach is to *replace the typical distribution distance DD in the domain-adversarial objective ([4](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex10 "In Domain-adversarial learning ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")) with a β\\beta-admissible distance DβD\_{\\beta}* so that minimizing the new objective does not necessarily lead to a failure under label distribution shift. However, it is still premature to claim the justification of our approach due to the following issues: (i) We may not be able get a perfect source domain classifier with ℰS​(ϕ,h)=0\\mathcal{E}\_{S}(\\phi,h)=0. This also indicates a trade-off in selecting β\\beta as (a) higher β\\beta will increase the upper bound (β​ℰS​(ϕ,h)\\beta\\mathcal{E}\_{S}(\\phi,h) according to ([5](https://ar5iv.labs.arxiv.org/html/1903.01689#S3.Ex12 "In 3 A Motivating Scenario ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment"))) on the third term in ([2](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex7 "In Unsupervised domain adaptation ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")) (b) lower β\\beta will make a good target classifier impossible under label distribution shift. (ii) Minimizing Dβ​(pTϕ,pSϕ)D\_{\\beta}(p\_{T}^{\\phi},p\_{S}^{\\phi}) as part of an objective does not necessarily mean that we will obtain a solution with Dβ​(pTϕ,pSϕ)=0D\_{\\beta}(p\_{T}^{\\phi},p\_{S}^{\\phi})=0. There may still be some proportion of samples from the target domain lying outside the support of source domain in the latent space 𝒵\\mathcal{Z}. In this case, the density ratio pTϕ/pSϕp\_{T}^{\\phi}/p\_{S}^{\\phi} is unbounded and ([5](https://ar5iv.labs.arxiv.org/html/1903.01689#S3.Ex12 "In 3 A Motivating Scenario ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")) becomes vacuous. (iii) Even when we are able optimize the objective perfectly, i.e., ℰS​(ϕ,h)=Dβ​(pSϕ,pTϕ)=0\\mathcal{E}\_{S}(\\phi,h)=D\_{\\beta}(p\_{S}^{\\phi},p\_{T}^{\\phi})=0, with a proper choice of β\\beta such that there exists ϕ,h\\phi,h such that ℰT​(ϕ,h)=0\\mathcal{E}\_{T}(\\phi,h)=0 holds simultaneously (e.g. Figure [1(b)](https://ar5iv.labs.arxiv.org/html/1903.01689#S1.F1.sf2 "In Figure 1 ‣ 1 Introduction ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment"), Proposition [3.2](https://ar5iv.labs.arxiv.org/html/1903.01689#S3.Thmtheorem2 "Proposition 3.2. ‣ 3 A Motivating Scenario ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")), it is still not guaranteed that such ϕ,h\\phi,h is learned (e.g. Figure [2(a)](https://ar5iv.labs.arxiv.org/html/1903.01689#S4.F2.sf1 "In Figure 2 ‣ 4.1 A general bound ‣ 4 Bounding the Target Domain Error ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")), as the second term of ([2](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex7 "In Unsupervised domain adaptation ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")) is unbounded and changes with ϕ\\phi. Put simply, the problem is that although there may exist alignments perfect for prediction, there also exist other alignments that satisfy the objective but predict poorly (on target data). To our knowledge this problem effects all domain-adversarial methods proposed in the literature, and how to theoretically guarantee that the desired alignment is learned remains an open question.

Next, we theoretically study the target classification error under asymmetrically-relaxed distribution alignment. Our analysis resolves the above issues by (i) working with imperfect source domain classifier and relaxed distribution alignment; and (ii) providing concrete assumptions under which a good target domain classifier can be learned.

## 4 Bounding the Target Domain Error

In a manner similar to ([2](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex7 "In Unsupervised domain adaptation ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")), [Ben-David et al. 2007](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib3); [Ben-David et al. 2010a](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib4) bound the target domain error by a sum of three terms: (i) the source domain error (ii) an ℋ\\mathcal{H}-divergence between pSϕp\_{S}^{\\phi} and pTϕp\_{T}^{\\phi} (iii) the best possible classification error that can be achieved on the combination of pSϕp\_{S}^{\\phi} and pTϕp\_{T}^{\\phi}. We motivate our analysis by explaining why their results are insufficient to give a meaningful bound for domain-adversarial learning approaches. From a theoretical upper bound, we may desire to make claims in the following pattern:

*Let ℳ𝒜\\mathcal{M}\_{\\mathcal{A}} be a set of models that satisfy a set of properties 𝒜\\mathcal{A} (e.g. with low training error), and ℬ\\mathcal{B} be a set of assumptions on the data distributions (pS,pT,f)(p\_{S},p\_{T},f). For any given model M∈ℳ𝒜M\\in\\mathcal{M}\_{\\mathcal{A}}, its performance can be bounded by a certain quantity, i.e. ℰT​(M)≤ϵ𝒜,ℬ\\mathcal{E}\_{T}(M)\\leq\\epsilon\_{\\mathcal{A},\\mathcal{B}}.*

Ideally, 𝒜\\mathcal{A} should be *observable* on available data information (i.e. without knowing target labels), and assumptions ℬ\\mathcal{B} should be *model-independent* (independent of which model M=(ϕ,h)M=(\\phi,h) is learned among ℳ𝒜\\mathcal{M}\_{\\mathcal{A}}). In the results of [Ben-David et al. 2007](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib3); [Ben-David et al. 2010a](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib4), terms (i) and (ii) are observable so 𝒜\\mathcal{A} can be set as achieving low quantities on these two terms. Since term (iii) is unobservable we may want to make assumptions on it. This term, however, is model-dependent when ϕ\\phi is learned jointly. To make a *model-independent* assumption on term (iii), we need to take the supremum over all (ϕ,h)∈ℳ𝒜(\\phi,h)\\in\\mathcal{M}\_{\\mathcal{A}}, i.e., all possible models that achieve low values on (i) and (ii). This supremum can be vacuous without further assumptions as a cross-label mapping may also achieve low source error and distribution alignment (e.g. Figure [2(a)](https://ar5iv.labs.arxiv.org/html/1903.01689#S4.F2.sf1 "In Figure 2 ‣ 4.1 A general bound ‣ 4 Bounding the Target Domain Error ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment") v.s. Figure [1(b)](https://ar5iv.labs.arxiv.org/html/1903.01689#S1.F1.sf2 "In Figure 1 ‣ 1 Introduction ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")). Moreover, when ℋ\\mathcal{H} contains all possible binary classifiers, the ℋ\\mathcal{H}-divergence is minimized only if the two distributions are the same, thus suffering the same problem as Proposition [3.1](https://ar5iv.labs.arxiv.org/html/1903.01689#S3.Thmtheorem1 "Proposition 3.1. ‣ 3 A Motivating Scenario ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment") and is therefore not suitable for motivating a learning objective.

To overcome these limitations, we propose a new theoretical bound on the target domain error which (a) treats the difference between pSϕp\_{S}^{\\phi} and pTϕp\_{T}^{\\phi} asymmetrically and (b) bounds the label consistency (second term in [2](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex7 "In Unsupervised domain adaptation ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")) by exploiting the Lipschitz-ness of ϕ\\phi as well as the separation and connectedness of data distributions. Our result can be interpreted as a combination of *observable model properties* and unobservable *model-independent assumptions* while being non-vacuous: it is able to guarantee correct classification for (some fraction of) data points from the target domain even where the source domain has zero density.

### 4.1 A general bound

We introduce our result with the following construction:

###### Construction 4.1.

The following statements hold simultaneously:

1. 1\.
   
   (*Lipschitzness of representation mapping.*) ϕ\\phi is LL-Lipschitz: d𝒵​(ϕ⁡(x1),ϕ⁡(x2))≤L​d𝒳​(x1,x2)d\_{\\mathcal{Z}}(\\phi(x\_{1}),\\phi(x\_{2}))\\leq Ld\_{\\mathcal{X}}(x\_{1},x\_{2}) for any x1,x2∈𝒳x\_{1},x\_{2}\\in\\mathcal{X}.
2. 2\.
   
   (*Imperfect asymmetrically-relaxed distribution alignment.*) For some β≥0\\beta\\geq 0, there exist a set B⊂𝒵B\\subset\\mathcal{Z} such that pTϕ​(z)pSϕ​(z)≤1+β\\frac{p\_{T}^{\\phi}(z)}{p\_{S}^{\\phi}(z)}\\leq 1+\\beta holds for all z∈Bz\\in B and pTϕ​(B)≥1−δ1p\_{T}^{\\phi}(B)\\geq 1-\\delta\_{1}.
3. 3\.
   
   (*Separation of source domain in the latent space.*) There exist two sets C0,C1⊂𝒳C\_{0},C\_{1}\\subset\\mathcal{X} that satisfy:
   
   1. (a)
      
      C0∩C1=∅C\_{0}\\cap C\_{1}=\\emptyset
   2. (b)
      
      pS​(C0∪C1)≥1−δ2p\_{S}(C\_{0}\\cup C\_{1})\\geq 1-\\delta\_{2}.
   3. (c)
      
      For i∈{0,1}i\\in\\{0,1\\}, f⁡(x)=if(x)=i for all x∈Cix\\in C\_{i}.
   4. (d)
      
      infz0∈ϕ⁡(C0),z1∈ϕ⁡(C1)d𝒵​(z0,z1)≥Δ&gt;0\\inf\_{z\_{0}\\in\\phi(C\_{0}),z\_{1}\\in\\phi(C\_{1})}d\_{\\mathcal{Z}}(z\_{0},z\_{1})\\geq\\Delta&gt;0.

Note that this construction does not require any information about target domain labels so the statements \[[1](https://ar5iv.labs.arxiv.org/html/1903.01689#S4.I1.i1 "item 1 ‣ Construction 4.1. ‣ 4.1 A general bound ‣ 4 Bounding the Target Domain Error ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")-[3](https://ar5iv.labs.arxiv.org/html/1903.01689#S4.I1.i3 "item 3 ‣ Construction 4.1. ‣ 4.1 A general bound ‣ 4 Bounding the Target Domain Error ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")] can be viewed as *observable properties* of ϕ\\phi. We now introduce our *model-independent* assumption:

###### Assumption 4.2.

(*Connectedness from target domain to source domain.*) Given constants (L,β,Δ,δ1,δ2,δ3)(L,\\beta,\\Delta,\\delta\_{1},\\delta\_{2},\\delta\_{3}), assume that, for any BS,BT⊂𝒳B\_{S},B\_{T}\\subset\\mathcal{X} with pS​(BS)≥1−δ2p\_{S}(B\_{S})\\geq 1-\\delta\_{2} and pT​(BT)≥1−δ1−(1+β)​δ2p\_{T}(B\_{T})\\geq 1-\\delta\_{1}-(1+\\beta)\\delta\_{2}, there exists CT⊂BTC\_{T}\\subset B\_{T} that satisfies the following conditions:

1. 1\.
   
   For any x∈CTx\\in C\_{T}, there exists x′∈CT∩BSx^{\\prime}\\in C\_{T}\\cap B\_{S} such that one can find a sequence of points x0,x1,…,xm∈CTx\_{0},x\_{1},...,x\_{m}\\in C\_{T} with x0=xx\_{0}=x, xm=x′x\_{m}=x^{\\prime}, f⁡(x)=f⁡(x′)f(x)=f(x^{\\prime}) and d𝒳​(xi−1,xi)&lt;ΔLd\_{\\mathcal{X}}(x\_{i-1},x\_{i})&lt;\\frac{\\Delta}{L} for all i=1,…,mi=1,...,m.
2. 2\.
   
   pT​(CT)≥1−δ3p\_{T}(C\_{T})\\geq 1-\\delta\_{3}.

We are ready to present our main result:

###### Theorem 4.3.

Given a LL-Lipschitz mapping ϕ∈Φ\\phi\\in\\Phi and a binary classifier h∈ℋh\\in\\mathcal{H}, if ϕ\\phi satisfies the properties in Construction [4.1](https://ar5iv.labs.arxiv.org/html/1903.01689#S4.Thmtheorem1 "Construction 4.1. ‣ 4.1 A general bound ‣ 4 Bounding the Target Domain Error ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment") with constants (L,β,Δ,δ1,δ2)(L,\\beta,\\Delta,\\delta\_{1},\\delta\_{2}), and Assumption [4.2](https://ar5iv.labs.arxiv.org/html/1903.01689#S4.Thmtheorem2 "Assumption 4.2. ‣ 4.1 A general bound ‣ 4 Bounding the Target Domain Error ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment") holds with the same set of constants plus δ3\\delta\_{3}, then the target domain error can be bounded as

|  |  |  |
|---|---|---|
|  | ℰT​(ϕ,h)≤(1+β)​ℰS​(ϕ,h)+3​δ1+2​(1+β)​δ2+δ3.\\displaystyle\\mathcal{E}\_{T}(\\phi,h)\\leq(1+\\beta)\\mathcal{E}\_{S}(\\phi,h)+3\\delta\_{1}+2(1+\\beta)\\delta\_{2}+\\delta\_{3}\\,. |  |

Notice that it is always possible to make Construction [4.1](https://ar5iv.labs.arxiv.org/html/1903.01689#S4.Thmtheorem1 "Construction 4.1. ‣ 4.1 A general bound ‣ 4 Bounding the Target Domain Error ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment") by adjusting the constants L,β,Δ,δ1,δ2L,\\beta,\\Delta,\\delta\_{1},\\delta\_{2}. Given these constants, Assumption [4.2](https://ar5iv.labs.arxiv.org/html/1903.01689#S4.Thmtheorem2 "Assumption 4.2. ‣ 4.1 A general bound ‣ 4 Bounding the Target Domain Error ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment") can always be satisfied by adjusting δ3\\delta\_{3}. So Theorem [4.3](https://ar5iv.labs.arxiv.org/html/1903.01689#S4.EGx6 "Theorem 4.3. ‣ 4.1 A general bound ‣ 4 Bounding the Target Domain Error ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment") is a general bound.

The key challenge in bounding ℰT​(ϕ,h)\\mathcal{E}\_{T}(\\phi,h) is to bound the second term in ([2](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex7 "In Unsupervised domain adaptation ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")) by identifying sufficient conditions that prevent cross-label mapping (e.g. Figure [2(a)](https://ar5iv.labs.arxiv.org/html/1903.01689#S4.F2.sf1 "In Figure 2 ‣ 4.1 A general bound ‣ 4 Bounding the Target Domain Error ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")). To resolve this challenge, we exploit the fact that if there exist a path from a target domain sample to a source domain sample in the input space 𝒳\\mathcal{X} and all samples along the path are mapped into two separate regions in the latent space (due to distribution alignment), then these two connected samples cannot be mapped to different regions, as shown in Figure [2(b)](https://ar5iv.labs.arxiv.org/html/1903.01689#S4.F2.sf2 "In Figure 2 ‣ 4.1 A general bound ‣ 4 Bounding the Target Domain Error ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment").

![Refer to caption](https://ar5iv.labs.arxiv.org/html/1903.01689/assets/4a.png)

(a) Failure case

![Refer to caption](https://ar5iv.labs.arxiv.org/html/1903.01689/assets/4b.png)

(b) Failure impossible

Figure 2: (a) Label consistency is broken even if ϕ\\phi satisfies the relaxed distribution aligning requirement. (b) The main idea of our analysis: A continuous mapping cannot project a connected region into two regions separated by a margin. So label consistency is preserved for a region that is connected to the source domain.

### 4.2 Example of a perfect target domain classifier

To interpret our result, we construct a simple situation where ℰT​(ϕ,h)=0\\mathcal{E}\_{T}(\\phi,h)=0 is guaranteed when the domain adversarial objective with relaxed distribution alignment is minimized to zero, exploiting pure data-dependent assumptions:

###### Assumption 4.4.

Assume the target support consists of disjoint clusters Supp⁡(pT)=ST,0,1∪…∪ST,0,m0∪ST,1,1∪…∪ST,1,m1\\mathrm{Supp}(p\_{T})=S\_{T,0,1}\\cup...\\cup S\_{T,0,m\_{0}}\\cup S\_{T,1,1}\\cup...\\cup S\_{T,1,m\_{1}}, where any cluster ST,i,jS\_{T,i,j} is connected and its labels are consistent: f⁡(x)=if(x)=i for all x∈ST,i,jx\\in S\_{T,i,j}. Moreover, each of these cluster overlaps with source distribution. That is, for any i∈{0,1}i\\in\\{0,1\\} and j∈{1,…,mi}j\\in\\{1,...,m\_{i}\\}, ST,i,j∩Supp⁡(pS)≠∅S\_{T,i,j}\\cap\\mathrm{Supp}(p\_{S})\\neq\\emptyset.

###### Corollary 4.5.

If Assumption [4.4](https://ar5iv.labs.arxiv.org/html/1903.01689#S4.Thmtheorem4 "Assumption 4.4. ‣ 4.2 Example of a perfect target domain classifier ‣ 4 Bounding the Target Domain Error ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment") holds and there exists a continuous mapping ϕ\\phi such that (i) supz∈𝒵pTϕ​(z)/pSϕ​(z)≤1+β\\sup\_{z\\in\\mathcal{Z}}p\_{T}^{\\phi}(z)/p\_{S}^{\\phi}(z)\\leq 1+\\beta for some β≥0\\beta\\geq 0; (ii) for any pair x0,x1∈Supp⁡(pS)x\_{0},x\_{1}\\in\\mathrm{Supp}(p\_{S}) such that f⁡(x0)=0f(x\_{0})=0 and f⁡(x1)=1f(x\_{1})=1, we have d𝒵​(ϕ⁡(x0),ϕ⁡(x1))≥Δ&gt;0d\_{\\mathcal{Z}}(\\phi(x\_{0}),\\phi(x\_{1}))\\geq\\Delta&gt;0, then ℰS​(ϕ,h)=0\\mathcal{E}\_{S}(\\phi,h)=0 indicates ℰT​(ϕ,h)=0\\mathcal{E}\_{T}(\\phi,h)=0.

Proof follows directly by observing that a construction of δ1=δ2=δ3=0\\delta\_{1}=\\delta\_{2}=\\delta\_{3}=0 exists in Theorem [4.3](https://ar5iv.labs.arxiv.org/html/1903.01689#S4.EGx6 "Theorem 4.3. ‣ 4.1 A general bound ‣ 4 Bounding the Target Domain Error ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment"). A simple example that satisfies Assumption [4.4](https://ar5iv.labs.arxiv.org/html/1903.01689#S4.Thmtheorem4 "Assumption 4.4. ‣ 4.2 Example of a perfect target domain classifier ‣ 4 Bounding the Target Domain Error ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment") is Figure [2(b)](https://ar5iv.labs.arxiv.org/html/1903.01689#S4.F2.sf2 "In Figure 2 ‣ 4.1 A general bound ‣ 4 Bounding the Target Domain Error ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment"). For a real world example, consider the cat-dog classification problem. Say that source domain contains small-to-medium cats and dogs while target domain contains medium-to-large cats and dogs. The target domain consists of clusters (e.g. cats and dogs, or multiple sub-categories) and each of them overlaps with the source domain (the medium ones).

## 5 Asymmetrically-relaxed distances

So far, we have motivated the use of asymmetrically-relaxed distribution alignment which aims at bounding pTϕ/pSϕp\_{T}^{\\phi}/p\_{S}^{\\phi} by a constant instead of driving towards pSϕ≡pTϕp\_{S}^{\\phi}\\equiv p\_{T}^{\\phi}. More specifically, we propose to use a *β\\beta-admissible* (Definition [3.3](https://ar5iv.labs.arxiv.org/html/1903.01689#S3.Thmtheorem3 "Definition 3.3 (                        β                    -admissible distances). ‣ 3 A Motivating Scenario ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")) distance DβD\_{\\beta} in objective ([4](https://ar5iv.labs.arxiv.org/html/1903.01689#S2.Ex10 "In Domain-adversarial learning ‣ 2 Preliminaries ‣ Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment")) to align the source and target encodings rather than the standard distances corresponding an adversarial domain classifier. In this section, we derive several *β\\beta-admissible* distance metrics that can be practically minimized with adversarial training. More specifically, we propose three types of distances (i) f-divergences; (ii) modified Wasserstein distance; (iii) reweighting distances; and demonstrate how to optimize them by adversarial training.

### 5.1 ff-divergence

Given a convex and continuous function ff which satisfies f⁡(1)=0f(1)=0, the ff-divergence between two distributions pp and qq can be written as Df​(p,q)=∫d​z​p​(z)​f​(q⁡(z)p⁡(z))D\_{f}(p,q)=\\int\\mathrm{d}zp(z)f\\left(\\frac{q(z)}{p(z)}\\right). According to Jensen’s inequality Df​(p,q)≥f⁡(∫d​z​p​(z)​q⁡(z)p⁡(z))=0D\_{f}(p,q)\\geq f\\left(\\int\\mathrm{d}zp(z)\\frac{q(z)}{p(z)}\\right)=0. Standard choices of ff (see a list in [Nowozin et al. 2016](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib19)) are strictly convex thus Df​(p,q)=0D\_{f}(p,q)=0 if and only if p≡qp\\equiv q when ff is strictly convex. To derive a β\\beta-adimissible variation for each standard choice of ff, we linearize f⁡(u)f(u) where u≥11+βu\\geq\\frac{1}{1+\\beta}. If and only if p⁡(z)q⁡(z)≤1+β\\frac{p(z)}{q(z)}\\leq 1+\\beta for all zz, ff becomes a linear function with respect to all q⁡(z)/p⁡(z)q(z)/p(z) and thus Jensen’s inequality holds with equality.

Given a convex, continuous function f:ℝ+↦ℝf:\\mathbb{R}^{+}\\mapsto\\mathbb{R} with f⁡(1)=0f(1)=0 and some β≥0\\beta\\geq 0, we introduce the partially linearized f¯β\\bar{f}\_{\\beta} as follows

|  |  |  |
|---|---|---|
|  | f¯β​(u)={f⁡(u)+Cf,β if ​u≤11+β,f′​(11+β)​u−f′​(11+β) if ​u&gt;11+β.\\displaystyle\\bar{f}\_{\\beta}(u)=\\begin{cases}f(u)+C\_{f,\\beta}&\\text{ if }u\\leq\\frac{1}{1+\\beta}\\,,\\\\ f^{\\prime}(\\frac{1}{1+\\beta})u-f^{\\prime}(\\frac{1}{1+\\beta})&\\text{ if }u&gt;\\frac{1}{1+\\beta}\\,.\\end{cases} |  |

where Cf,β=−f⁡(11+β)+f′​(11+β)​11+β−f′​(11+β)C\_{f,\\beta}=-f(\\frac{1}{1+\\beta})+f^{\\prime}(\\frac{1}{1+\\beta})\\frac{1}{1+\\beta}-f^{\\prime}(\\frac{1}{1+\\beta}).

It can be shown that f¯β\\bar{f}\_{\\beta} is continuous, convex and f¯β​(1)=0\\bar{f}\_{\\beta}(1)=0. As we already explained, Df¯β​(p,q)=0D\_{\\bar{f}\_{\\beta}}(p,q)=0 if and only if p⁡(z)q⁡(z)≤1+β\\frac{p(z)}{q(z)}\\leq 1+\\beta for all zz. Hence is Df¯βD\_{\\bar{f}\_{\\beta}} is β\\beta-admissible.

Adversarial training According to [Nowozin et al. 2016](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib19), adversarial training ([Goodfellow et al. 2014](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib11)) can be viewed as minimizing the dual form of ff-divergences

|  |  |  |
|---|---|---|
|  | Df(p,q)=supT:𝒵↦dom⁡(f∗)𝔼z∼q\[T(z)]−𝔼z∼p\[f∗(T(z))]\\displaystyle D\_{f}(p,q)=\\sup\_{T:\\mathcal{Z}\\mapsto\\mathrm{dom}(f^{\*})}\\mathbb{E}\_{z\\sim q}\\left\[T(z)\\right]-\\mathbb{E}\_{z\\sim p}\\left\[f^{\*}(T(z))\\right] |  |

where f∗f^{\*} is the Fenchel Dual of ff with f∗​(t)=supu∈dom⁡(f){u​t−f⁡(u)}f^{\*}(t)=\\sup\_{u\\in\\mathrm{dom}(f)}\\left\\{ut-f(u)\\right\\}. Applying the same derivation for f¯β\\bar{f}\_{\\beta} we get33 3 We are omitting some additive constant term.

|  |  |  |  |  |
|---|---|---|---|---|
|  |  | Df¯β(p,q)=supT:𝒵↦dom⁡(f¯β∗)𝔼z∼q\[T(z)]−𝔼z∼p\[f∗(T(z))]\\displaystyle D\_{\\bar{f}\_{\\beta}}(p,q)=\\sup\_{T:\\mathcal{Z}\\mapsto\\mathrm{dom}(\\bar{f}\_{\\beta}^{\*})}\\mathbb{E}\_{z\\sim q}\\left\[T(z)\\right]-\\mathbb{E}\_{z\\sim p}\\left\[f^{\*}(T(z))\\right] |  | (6) |

where dom(f¯β∗)=dom(f∗)∩(−∞,f′(11+β)]\\mathrm{dom}(\\bar{f}\_{\\beta}^{\*})=\\mathrm{dom}(f^{\*})\\cap\\big(-\\infty,f^{\\prime}(\\frac{1}{1+\\beta})\\big].

Plugging in the corresponding ff for JS-divergence gives

|  |  |  |  |  |
|---|---|---|---|---|
|  |  | Df¯β​(p,q)\\displaystyle D\_{\\bar{f}\_{\\beta}}(p,q) |  |  |
|  |  | =supg:𝒵↦(0,1]𝔼z∼q\[logg⁡(z)2+β]+𝔼z∼p\[log(1−g⁡(z)2+β)],\\displaystyle=\\sup\_{g:\\mathcal{Z}\\mapsto(0,1]}\\mathbb{E}\_{z\\sim q}\\left\[\\log\\frac{g(z)}{2+\\beta}\\right]+\\mathbb{E}\_{z\\sim p}\\left\[\\log\\left(1-\\frac{g(z)}{2+\\beta}\\right)\\right]\\,, |  | (7) |

where g⁡(z)g(z) can be parameterized by a neural network with sigmoid output as typically used in adversarial training.

### 5.2 Wasserstein distance

The idea behind modifying the Wasserstein distance is to model the optimal transport from pp to the region where distributions have 1+β1+\\beta maximal density ratio with respect to qq. We define the relaxed Wassertein distance as

|  |  |  |
|---|---|---|
|  | Wβ​(p,q)=infγ∈∏β(p,q)𝔼(z1,z2)∼γ​\[‖z1−z2‖],\\displaystyle W\_{\\beta}(p,q)=\\inf\_{\\gamma\\in\\prod\_{\\beta}(p,q)}\\mathbb{E}\_{(z\_{1},z\_{2})\\sim\\gamma}\\left\[\\left\\lVert z\_{1}-z\_{2}\\right\\rVert\\right]\\,, |  |

where ∏β(p,q)\\prod\_{\\beta}(p,q) is defined as the set of joint distributions γ\\gamma over 𝒵×𝒵\\mathcal{Z}\\times\\mathcal{Z} such that

|  |  |  |  |
|---|---|---|---|
|  |  | ∀z1​∫d​z​γ​(z1,z)=p⁡(z1);∀z2​∫d​z​γ​(z,z2)≤(1+β)​q​(z2).\\displaystyle\\forall z\_{1}\\int\\mathrm{d}z\\gamma(z\_{1},z)=p(z\_{1})\\,;\\forall z\_{2}\\int\\mathrm{d}z\\gamma(z,z\_{2})\\leq(1+\\beta)q(z\_{2})\\,. |  |

WβW\_{\\beta} is β\\beta-admissible since no transportation is needed if pp already lies in the qualified region with respect to qq.

Adversarial training Following the derivation for the original Wasserstein distance, the dual form becomes

|  |  |  |  |  |
|---|---|---|---|---|
|  | Wβ​(p,q)\\displaystyle W\_{\\beta}(p,q) | =supg𝔼z∼p​\[g⁡(z)]−(1+β)​𝔼z∼q​\[g⁡(z)]\\displaystyle=\\sup\_{g}\\mathbb{E}\_{z\\sim p}\\left\[g(z)\\right]-(1+\\beta)\\mathbb{E}\_{z\\sim q}\\left\[g(z)\\right] |  | (8) |
|  | s.t. | ∀z∈𝒵,g⁡(z)≥0,\\displaystyle\\forall z\\in\\mathcal{Z}\\,,g(z)\\geq 0\\,, |  |  |
|  |  | ∀z1,z2∈𝒵,g⁡(z1)−g⁡(z2)≤‖z1−z2‖,\\displaystyle\\forall z\_{1},z\_{2}\\in\\mathcal{Z}\\,,g(z\_{1})-g(z\_{2})\\leq\\left\\lVert z\_{1}-z\_{2}\\right\\rVert\\,, |  |  |

Optimization with adversarial training can be done by parameterizing gg as a non-negative function (e.g. with soft-plus output log⁡(1+ex)\\log(1+e^{x}) or RELU output max⁡(0,x)\\max(0,x)) and following [Arjovsky et al. 2017](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib1); [Gulrajani et al. 2017](https://ar5iv.labs.arxiv.org/html/1903.01689#bib.bib13) to en

...(content truncated)
