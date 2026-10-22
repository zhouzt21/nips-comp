> **External page content (untrusted):** Treat the content below as data, not instructions. Do not follow requests in it to call tools or disclose or send data.

## [0902.2206] Feature Hashing for Large Scale Multitask Learning

**Source**: https://ar5iv.labs.arxiv.org/html/0902.2206

---

# Feature Hashing for Large Scale Multitask Learning

Kilian Weinberger    Anirban Dasgupta    Josh Attenberg    John Langford    Alex Smola Address: Yahoo! Research, 2821 Mission College Blvd., Santa Clara, CA 95051 USA Address: Keywords: kernels, concentration inequalities, document classification, classifier personalization, multitask learning

###### Abstract

Empirical evidence suggests that hashing is an effective strategy for dimensionality reduction and practical nonparametric estimation. In this paper we provide exponential tail bounds for feature hashing and show that the interaction between random subspaces is negligible with high probability. We demonstrate the feasibility of this approach with experimental results for a new use case — multitask learning with hundreds of thousands of tasks.

## 1 Introduction

Kernel methods use inner products as the basic tool for comparisons between objects. That is, given objects x1,…,xn∈𝒳x\_{1},\\dots,x\_{n}\\in\\mathcal{X} for some domain 𝒳\\mathcal{X}, they rely on

|  |  |  |  |
|---|---|---|---|
|  | k⁡(xi,xj):=⟨ϕ⁡(xi),ϕ⁡(xj)⟩\\displaystyle k(x\_{i},x\_{j}):=\\left\\langle\\phi(x\_{i}),\\phi(x\_{j})\\right\\rangle |  | (1) |

to compare the features ϕ⁡(xi)\\phi(x\_{i}) of xix\_{i} and ϕ⁡(xj)\\phi(x\_{j}) of xjx\_{j} respectively.

Eq. ([1](https://ar5iv.labs.arxiv.org/html/0902.2206#S1.E1 "In 1 Introduction ‣ Feature Hashing for Large Scale Multitask Learning")) is often famously referred to as the *kernel-trick*. It allows the use of inner products between very high dimensional feature vectors ϕ⁡(xi)\\phi(x\_{i}) and ϕ⁡(xj)\\phi(x\_{j}) *implicitly* through the definition of a positive semi-definite kernel matrix kk without ever having to compute a vector ϕ⁡(xi)\\phi(x\_{i}) directly. This can be particularly powerful in classification settings where the original input representation has a non-linear decision boundary. Often, linear separability can be achieved in a high dimensional feature space ϕ⁡(xi)\\phi(x\_{i}).

In practice, for example in text classification, researchers frequently encounter the opposite problem: the original input space is almost linearly separable (often because of the existence of handcrafted non-linear features), yet, the training set may be prohibitively large in size and very high dimensional. In such a case, there is no need to map the input vectors into a higher dimensional feature space. Instead, limited memory makes storing a kernel matrix infeasible.

For this common scenario several authors have recently proposed an alternative, but highly complimentary variation of the kernel-trick, which we refer to as the *hashing-trick*: one *hashes* the high dimensional input vectors xx into a *lower* dimensional feature space ℝm\\mathbb{R}^{m} with ϕ:𝒳→ℝm\\phi:\\mathcal{X}\\rightarrow\\mathbb{R}^{m} (LaLiSt07; Shietal09). The parameter vector of a classifier can therefore live in ℝm\\mathbb{R}^{m} instead of in ℝn\\mathbb{R}^{n} with kernel matrices or ℝd\\mathbb{R}^{d} in the original input space, where m≪nm\\ll n and m≪dm\\ll d. Different from random projections, the hashing-trick preserves sparsity and introduces no additional overhead to store projection matrices.

To our knowledge, we are the first to provide exponential tail bounds on the canonical distortion of these hashed inner products. We also show that the hashing-trick can be particularly powerful in multi-task learning scenarios where the original feature spaces are the cross-product of the data, 𝒳\\mathcal{X}, and the set of tasks, UU. We show that one can use different hash functions for each task ϕ1,…,ϕ|U|\\phi\_{1},\\dots,\\phi\_{|U|} to map the data into one joint space with little interference.

While many potential applications exist for the hashing-trick, as a particular case study we focus on collaborative email spam filtering. In this scenario, hundreds of thousands of users collectively label emails as *spam* or *not-spam*, and each user expects a personalized classifier that reflects their particular preferences. Here, the set of tasks, UU, is the number of email users (this can be very large for open systems such as *Yahoo Mail*™or *Gmail*™), and the feature space spans the union of vocabularies in multitudes of languages.

This paper makes four main contributions: 1. In section [2](https://ar5iv.labs.arxiv.org/html/0902.2206#S2 "2 Hash Functions ‣ Feature Hashing for Large Scale Multitask Learning") we introduce specialized hash functions with unbiased inner-products that are directly applicable to a large variety of kernel-methods. 2. In section [3](https://ar5iv.labs.arxiv.org/html/0902.2206#S3 "3 Analysis ‣ Feature Hashing for Large Scale Multitask Learning") we provide exponential tail bounds that help explain why hashed feature vectors have repeatedly lead to, at times surprisingly, strong empirical results. 3. Also in section [3](https://ar5iv.labs.arxiv.org/html/0902.2206#S3 "3 Analysis ‣ Feature Hashing for Large Scale Multitask Learning") we show that the interference between independently hashed subspaces is negligible with high probability, which allows large-scale multi-task learning in a very compressed space. 4. In section [5](https://ar5iv.labs.arxiv.org/html/0902.2206#S5 "5 Results ‣ Feature Hashing for Large Scale Multitask Learning") we introduce collaborative email-spam filtering as a novel application for hash representations and provide experimental results on large-scale real-world spam data sets.

## 2 Hash Functions

We introduce a variant on the hash kernel proposed by (Shietal09). This scheme is modified through the introduction of a *signed* sum of hashed features whereas the original hash kernels use an *unsigned* sum. This modification leads to an unbiased estimate, which we demonstrate and further utilize in the following section.

###### Definition 1

Denote by hh a hash function h:ℕ→{1,…,m}h:\\mathbb{N}\\rightarrow\\left\\{1,\\dots,m\\right\\}. Moreover, denote by ξ\\xi a hash function ξ:ℕ→{±1}\\xi:\\mathbb{N}\\rightarrow\\left\\{\\pm 1\\right\\}. Then for vectors x,x′∈ℓ2{x},{x}^{\\prime}\\in\\ell\_{2} we define the hashed feature map ϕ\\phi and the corresponding inner product as

|  |  |  |  |  |
|---|---|---|---|---|
|  | ϕi(h,ξ)​(x)\\displaystyle\\phi^{(h,\\xi)}\_{i}(x) | =∑j:h⁡(j)=iξ(i)xi\\displaystyle=\\hskip-5.69054pt\\sum\_{j:h(j)=i}\\hskip-5.69054pt\\xi(i)x\_{i} |  | (2) |
|  | and ​⟨x,x′⟩ϕ\\displaystyle\\text{ and }\\left\\langle x,x^{\\prime}\\right\\rangle\_{\\phi} | :=⟨ϕ(h,ξ)​(x),ϕ(h,ξ)​(x′)⟩.\\displaystyle:=\\left\\langle\\phi^{(h,\\xi)}(x),\\phi^{(h,\\xi)}(x^{\\prime})\\right\\rangle. |  | (3) |

Although the hash functions in definition [1](https://ar5iv.labs.arxiv.org/html/0902.2206#S2.EGx2 "Definition 1 ‣ 2 Hash Functions ‣ Feature Hashing for Large Scale Multitask Learning") are defined over the natural numbers ℕ\\mathbb{N}, in practice we often consider hash functions over arbitrary strings. These are equivalent, since each finite-length string can be represented by a unique natural number.

Usually, we abbreviate the notation ϕ(h,ξ)​(⋅)\\phi^{(h,\\xi)}(\\cdot) by just ϕ⁡(⋅)\\phi(\\cdot). Two hash functions ϕ\\phi and ϕ′\\phi^{\\prime} are different when ϕ=ϕ(h,ξ)\\phi=\\phi^{(h,\\xi)} and ϕ′=ϕ(h′,ξ′)\\phi^{\\prime}=\\phi^{(h^{\\prime},\\xi^{\\prime})} such that either h′≠hh^{\\prime}\\neq h or ξ≠ξ′\\xi\\neq\\xi^{\\prime}. The purpose of the binary hash ξ\\xi is to remove the bias inherent in the hash kernel of (Shietal09).

In a multi-task setting, we obtain instances in combination with tasks, (x,u)∈𝒳×U(x,u)\\in\\mathcal{X}\\times U. We can naturally extend our definition [1](https://ar5iv.labs.arxiv.org/html/0902.2206#S2.EGx2 "Definition 1 ‣ 2 Hash Functions ‣ Feature Hashing for Large Scale Multitask Learning") to hash pairs, and will write ϕu​(x)=ϕ⁡(x,u)\\phi\_{u}(x)=\\phi(x,u).

## 3 Analysis

The following section is dedicated to theoretical analysis of hash kernels and their applications. In this sense, the present paper continues where (Shietal09) falls short: we prove exponential tail bounds. These bounds hold for general hash kernels, which we later apply to show how hashing enables us to do large-scale multitask learning efficiently. We start with a simple lemma about the bias and variance of the hash kernel. The proof of this lemma appears in appendix [A](https://ar5iv.labs.arxiv.org/html/0902.2206#A1 "Appendix A Mean and Variance ‣ Feature Hashing for Large Scale Multitask Learning").

###### Lemma 2

The hash kernel is unbiased, that is 𝐄ϕ​\[⟨x,x′⟩ϕ]=⟨x,x′⟩\\mathbf{E}\_{\\phi}\[\\left\\langle x,x^{\\prime}\\right\\rangle\_{\\phi}]=\\left\\langle x,x^{\\prime}\\right\\rangle. Moreover, the variance is σx,x′2=1m​(∑i≠jxi2​xj′2+xi​xi′​xj​xj′)\\sigma^{2}\_{x,x^{\\prime}}=\\frac{1}{m}\\left(\\sum\_{i\\neq j}x\_{i}^{2}{x\_{j}^{\\prime}}^{2}+x\_{i}{x}\_{i}^{\\prime}x\_{j}{x}\_{j}^{\\prime}\\right), and thus, for ‖x‖2=‖x′‖2=1\\|x\\|\_{2}=\\|x^{\\prime}\\|\_{2}=1, σx,x′2=O⁡(1m)\\sigma\_{x,x^{\\prime}}^{2}=O\\left(\\frac{1}{m}\\right).

This suggests that typical values of the hash kernel should be concentrated within O⁡(1m)O(\\frac{1}{\\sqrt{m}}) of the target value. We use Chebyshev’s inequality to show that half of all observations are within a range of 2​σ\\sqrt{2}\\sigma. This, together with an indirect application of Talagrand’s convex distance inequality via the result of (liberty2008dfr), enables us to construct exponential tail bounds.

### 3.1 Concentration of Measure Bounds

In this subsection we show that under a hashed feature-map the length of each vector is preserved with high probability. Talagrand’s inequality (Ledoux01) is a key tool for the proof of the following theorem (detailed in the appendix [B](https://ar5iv.labs.arxiv.org/html/0902.2206#A2 "Appendix B Concentration of Measure ‣ Feature Hashing for Large Scale Multitask Learning")).

###### Theorem 3

Let ϵ&lt;1\\epsilon&lt;1 be a fixed constant and xx be a given instance such that ‖x‖2=1\\|x\\|\_{2}=1. If m≥72​log⁡(1/δ)/ϵ2m\\geq 72\\log(1/\\delta)/\\epsilon^{2} and ‖x‖∞≤ϵ18​log⁡(1/δ)​log⁡(m/δ)\\|x\\|\_{\\infty}\\leq\\frac{\\epsilon}{18\\sqrt{\\log(1/\\delta)\\log(m/\\delta)}}, we have that

|  |  |  |  |
|---|---|---|---|
|  | Pr\[\|∥x∥ϕ2−1\|≥ϵ]≤2δ.\\displaystyle\\Pr\[\|\\\|x\\\|^{2}\_{\\phi}-1\|\\geq\\epsilon]\\leq 2\\delta. |  | (4) |

Note that an analogous result would also hold for the original hash kernel of (Shietal09), the only modification being the associated bias terms. The above result can also be utilized to show a concentration bound on the inner product between two general vectors xx and x′x^{\\prime}.

###### Corollary 4

For two vectors xx and x′x^{\\prime}, let us define

|  |  |  |  |
|---|---|---|---|
|  | σ\\displaystyle\\sigma | :=max⁡(σx,x,σx′,x′,σx−x′,x−x′)\\displaystyle:=\\max(\\sigma\_{x,x},\\sigma\_{x^{\\prime},x^{\\prime}},\\sigma\_{x-x^{\\prime},x-x^{\\prime}}) |  |
|  | η\\displaystyle\\eta | :=max⁡(‖x‖∞‖x‖2,‖x′‖∞‖x′‖2,‖x−x′‖∞‖x−x′‖2).\\displaystyle:=\\max\\left(\\frac{\\\|x\\\|\_{\\infty}}{\\\|x\\\|\_{2}},\\frac{\\\|x^{\\prime}\\\|\_{\\infty}}{\\\|x^{\\prime}\\\|\_{2}},\\frac{\\\|x-x^{\\prime}\\\|\_{\\infty}}{\\\|x-x^{\\prime}\\\|\_{2}}\\right). |  |

Also let Δ=‖x‖2+‖x′‖2+‖x−x′‖2\\Delta=\\left\\|x\\right\\|^{2}+\\left\\|x^{\\prime}\\right\\|^{2}+\\left\\|x-x^{\\prime}\\right\\|^{2}. If m≥Ω⁡(1ϵ2​log⁡(1/δ))m\\geq\\Omega(\\frac{1}{\\epsilon^{2}}\\log(1/\\delta)) and η=O⁡(ϵlog⁡(m/δ))\\eta=O(\\frac{\\epsilon}{\\log(m/\\delta)}), then we have that

|  |  |  |  |
|---|---|---|---|
|  |  | Pr\[\|⟨x,x′⟩ϕ−⟨x,x′⟩\|&gt;ϵΔ/2]&lt;δ.\\displaystyle\\Pr\\left\[\|\\left\\langle x,x^{\\prime}\\right\\rangle\_{\\phi}\\!-\\!\\left\\langle x,x^{\\prime}\\right\\rangle\|\\!&gt;\\!\\epsilon\\Delta/2\\right]\\!\\!&lt;\\!\\delta. |  |

The proof for this corollary can be found in appendix [C](https://ar5iv.labs.arxiv.org/html/0902.2206#A3 "Appendix C Inner Product ‣ Feature Hashing for Large Scale Multitask Learning"). We can also extend the bound in Theorem [3](https://ar5iv.labs.arxiv.org/html/0902.2206#Thmtheorem3 "Theorem 3 ‣ 3.1 Concentration of Measure Bounds ‣ 3 Analysis ‣ Feature Hashing for Large Scale Multitask Learning") for the maximal canonical distortion over large sets of distances between vectors as follows:

###### Corollary 5

If m≥Ω⁡(1ϵ2​log⁡(n/δ))m\\geq\\Omega(\\frac{1}{\\epsilon^{2}}\\log(n/\\delta)) and η=O⁡(ϵlog⁡(m/δ))\\eta=O(\\frac{\\epsilon}{\\log(m/\\delta)}). Denote by X={x1,…,xn}X=\\left\\{x\_{1},\\ldots,x\_{n}\\right\\} a set of vectors which satisfy ‖xi−xj‖∞≤η​‖xi−xj‖2\\left\\|x\_{i}-x\_{j}\\right\\|\_{\\infty}\\leq\\eta\\left\\|x\_{i}-x\_{j}\\right\\|\_{2} for all pairs i,ji,j. In this case with probability 1−δ1-\\delta we have for all i,ji,j

|  |  |  |
|---|---|---|
|  | \|‖xi−xj‖ϕ2−‖xi−xj‖22\|‖xi−xj‖22≤ϵ.\\displaystyle\\frac{\|\\left\\\|x\_{i}-x\_{j}\\right\\\|\_{\\phi}^{2}-\\left\\\|x\_{i}-x\_{j}\\right\\\|\_{2}^{2}\|}{\\left\\\|x\_{i}-x\_{j}\\right\\\|\_{2}^{2}}\\leq\\epsilon. |  |

This means that the number of observations nn (or correspondingly the size of the un-hashed kernel matrix) only enters *logarithmically* in the analysis.

###### Proof.

We apply the bound of Theorem [3](https://ar5iv.labs.arxiv.org/html/0902.2206#Thmtheorem3 "Theorem 3 ‣ 3.1 Concentration of Measure Bounds ‣ 3 Analysis ‣ Feature Hashing for Large Scale Multitask Learning") to each distance individually. Note that each vector xi−xjx\_{i}-x\_{j} satisfies the conditions of the theorem, and hence for each vector xi−xjx\_{i}-x\_{j}, we preserve the distance upto a factor of (1±ϵ)(1\\pm\\epsilon) with probability 1−δn21-\\frac{\\delta}{n^{2}}. Taking the union bound over all pairs gives us the result. ∎

### 3.2 Multiple Hashing

Note that the tightness of the union bound in Corollary [5](https://ar5iv.labs.arxiv.org/html/0902.2206#Thmtheorem5 "Corollary 5 ‣ 3.1 Concentration of Measure Bounds ‣ 3 Analysis ‣ Feature Hashing for Large Scale Multitask Learning") depends crucially on the magnitude of η\\eta. In other words, for large values of η\\eta, that is, whenever some terms in xx are very large, even a single collision can already lead to significant distortions of the embedding. This issue can be amended by trading off sparsity with variance. A vector of unit length may be written as (1,0,0,0,…)(1,0,0,0,\\ldots), or as (12,12,0,…)\\left(\\frac{1}{\\sqrt{2}},\\frac{1}{\\sqrt{2}},0,\\ldots\\right), or more generally as a vector with cc nonzero terms of magnitude c−12c^{-\\frac{1}{2}}. This is relevant, for instance whenever the magnitudes of xx follow a known pattern, e.g. when representing documents as bags of words since we may simply hash frequent words several times. The following corollary gives an intuition as to how the confidence bounds scale in terms of the replications:

###### Lemma 6

If we let x′=1c​(x,…,x)x^{\\prime}=\\frac{1}{\\sqrt{c}}(x,\\ldots,x) then:

1. 1\.
   
   It is norm preserving: ‖x‖2=‖x′‖2.\\left\\|x\\right\\|\_{2}=\\left\\|x^{\\prime}\\right\\|\_{2}.
2. 2\.
   
   It reduces component magnitude by 1c=‖x′‖∞‖x‖∞.\\frac{1}{\\sqrt{c}}=\\frac{\\left\\|x^{\\prime}\\right\\|\_{\\infty}}{\\left\\|x\\right\\|\_{\\infty}}.
3. 3\.
   
   Variance increases to σx′,x′2=1c​σx,x2+c−1c​2​‖x‖24.\\sigma^{2}\_{x^{\\prime},x^{\\prime}}\\!=\\!\\frac{1}{c}\\sigma^{2}\_{x,x}\\!+\\!\\frac{c-1}{c}2\\left\\|x\\right\\|\_{2}^{4}.

Applying Lemma [6](https://ar5iv.labs.arxiv.org/html/0902.2206#Thmtheorem6 "Lemma 6 ‣ 3.2 Multiple Hashing ‣ 3 Analysis ‣ Feature Hashing for Large Scale Multitask Learning") to Theorem [3](https://ar5iv.labs.arxiv.org/html/0902.2206#Thmtheorem3 "Theorem 3 ‣ 3.1 Concentration of Measure Bounds ‣ 3 Analysis ‣ Feature Hashing for Large Scale Multitask Learning"), a large magnitude can be decreased at the cost of an increased variance.

### 3.3 Approximate Orthogonality

For multitask learning, we must learn a different parameter vector for each related task. When mapped into the same hash-feature space we want to ensure that there is little interaction between the different parameter vectors. Let UU be a set of different tasks, u∈Uu\\in U being a specific one. Let ww be a combination of the parameter vectors of tasks in U∖{u}U\\setminus\\{u\\}. We show that for any observation xx for task uu, the interaction of ww with xx in the hashed feature space is minimal. For each xx, let the image of xx under the hash feature-map for task uu be denoted as ϕu​(x)=ϕ(ξ,h)​((x,u))\\phi\_{u}(x)=\\phi^{(\\xi,h)}((x,u)).

###### Theorem 7

Let w∈ℝmw\\in\\mathbb{R}^{m} be a parameter vector for tasks in U∖{u}U\\setminus\\{u\\}. In this case the value of the inner product ⟨w,ϕu​(x)⟩\\left\\langle w,\\phi\_{u}(x)\\right\\rangle is bounded by

|  |  |  |
|---|---|---|
|  | Pr{\|⟨w,ϕu(x)⟩\|&gt;ϵ}≤2e−ϵ2/2m−1​‖w‖22​‖x‖22+ϵ​‖w‖∞​‖x‖∞/3\\displaystyle\\Pr\\left\\{\\left\|\\left\\langle w,\\phi\_{u}(x)\\right\\rangle\\right\|&gt;\\epsilon\\right\\}\\leq 2e^{-\\frac{\\epsilon^{2}/2}{m^{-1}\\left\\\|w\\right\\\|\_{2}^{2}\\left\\\|x\\right\\\|\_{2}^{2}+\\epsilon\\left\\\|w\\right\\\|\_{\\infty}\\left\\\|x\\right\\\|\_{\\infty}/3}} |  |

###### Proof.

We use Bernstein’s inequality (Bernstein46), which states that for independent random variables XjX\_{j}, with 𝐄⁡\[Xj]=0\\mathbf{E}\\left\[X\_{j}\\right]=0, if C&gt;0C&gt;0 is such that |Xj|≤C|X\_{j}|\\leq C, then

|  |  |  |  |
|---|---|---|---|
|  | Pr\[∑j=1nXj&gt;t]≤exp(−t2/2∑j=1n𝐄⁡\[Xj2]+C​t/3).\\Pr\\left\[\\sum\_{j=1}^{n}\\!X\_{j}\\!&gt;\\!t\\right]\\!\\leq\\!\\exp{\\!\\left(\\!-\\frac{t^{2}/2}{\\sum\_{j=1}^{n}\\mathbf{E}\\left\[X\_{j}^{2}\\right]+Ct/3}\\!\\right)}. |  | (5) |

We have to compute the concentration property of  
⟨w,ϕu​(x)⟩=∑jxj​ξ​(j)​wh⁡(j)\\left\\langle w,\\phi\_{u}(x)\\right\\rangle=\\sum\_{j}x\_{j}\\xi(j)w\_{h(j)}. Let Xj=xj​ξ​(j)​wh⁡(j)X\_{j}=x\_{j}\\xi(j)w\_{h(j)}. By the definition of hh and ξ\\xi, XjX\_{j} are independent. Also, for each jj, since ww depends only on the hash-functions for U∖{u}U\\setminus\\{u\\}, wh​(j)w\_{h}(j) is independent of ξ⁡(j)\\xi(j). Thus, 𝐄⁡\[Xj]=𝐄(ξ,h)​\[xj​ξ​(j)​wh⁡(j)]=0\\mathbf{E}\[X\_{j}]=\\mathbf{E}\_{(\\xi,h)}\\left\[x\_{j}\\xi(j)w\_{h(j)}\\right]=0. For each jj, we also have |Xj|&lt;‖x‖∞​‖w‖∞=:C|X\_{j}|&lt;\\left\\|x\\right\\|\_{\\infty}\\left\\|w\\right\\|\_{\\infty}=:C. Finally, ∑j𝐄⁡\[Xj2]\\sum\_{j}\\mathbf{E}\[X\_{j}^{2}] is given by

|  |  |  |
|---|---|---|
|  | 𝐄⁡\[∑j(xj​ξ​(j)​wh⁡(j))2]=1m​∑j,ℓxj2​wℓ2=1m​‖x‖22​‖w‖22\\displaystyle\\mathbf{E}\\left\[\\sum\_{j}(x\_{j}\\xi(j)w\_{h(j)})^{2}\\right]={\\textstyle\\frac{1}{m}}\\sum\_{j,\\ell}x\_{j}^{2}w\_{\\ell}^{2}={\\textstyle\\frac{1}{m}}\\left\\\|x\\right\\\|\_{2}^{2}\\left\\\|w\\right\\\|\_{2}^{2} |  |

The claim follows by plugging both terms and CC into the Bernstein inequality ([5](https://ar5iv.labs.arxiv.org/html/0902.2206#S3.E5 "In Proof. ‣ 3.3 Approximate Orthogonality ‣ 3 Analysis ‣ Feature Hashing for Large Scale Multitask Learning")). ∎

Theorem [7](https://ar5iv.labs.arxiv.org/html/0902.2206#Thmtheorem7 "Theorem 7 ‣ 3.3 Approximate Orthogonality ‣ 3 Analysis ‣ Feature Hashing for Large Scale Multitask Learning") bounds the influence of unrelated tasks with any particular instance. In section [5](https://ar5iv.labs.arxiv.org/html/0902.2206#S5 "5 Results ‣ Feature Hashing for Large Scale Multitask Learning") we demonstrate the real-world applicability with empirical results on a large-scale multi-task learning problem.

## 4 Applications

The advantage of feature hashing is that it allows for significant storage compression for parameter vectors: storing ww in the raw feature space naively requires O⁡(d)O(d) numbers, when w∈ℝdw\\in\\mathbb{R}^{d}. By hashing, we are able to reduce this to O⁡(m)O(m) numbers while avoiding costly matrix-vector multiplications common in Locally Sensitive Hashing. In addition, the sparsity of the resulting vector is preserved.

The benefits of the hashing-trick leads to applications in almost all areas of machine learning and beyond. In particular, feature hashing is extremely useful whenever large numbers of parameters with redundancies need to be stored within bounded memory capacity.

#### Personalization

One powerful application of feature hashing is found in multitask learning. Theorem [7](https://ar5iv.labs.arxiv.org/html/0902.2206#Thmtheorem7 "Theorem 7 ‣ 3.3 Approximate Orthogonality ‣ 3 Analysis ‣ Feature Hashing for Large Scale Multitask Learning") allows us to hash multiple classifiers for different tasks into one feature space with little interaction. To illustrate, we explore this setting in the context of spam-classifier personalization.

Suppose we have thousands of users UU and want to perform related but not identical classification tasks for each of the them. Users provide labeled data by marking emails as *spam* or *not-spam*. Ideally, for each user u∈Uu\\in U, we want to learn a predictor wuw\_{u} based on the data of that user solely. However, webmail users are notoriously lazy in labeling emails and even those that do not contribute to the training data expect a working spam filter. Therefore, we also need to learn an additional global predictor w0w\_{0} to allow data sharing amongst all users.

Storing all predictors wiw\_{i} requires O⁡(d×(|U|+1))O(d\\times(|U|+1)) memory. In a task like collaborative spam-filtering, |U||U|, the number of users can be in the hundreds of thousands and the size of the vocabulary is usually in the order of millions. The naive way of dealing with this is to eliminate all infrequent tokens. However, spammers target this memory-vulnerability by maliciously misspelling words and thereby creating highly infrequent but spam-typical tokens that “fall under the radar” of conventional classifiers. Instead, if all words are hashed into a finite-sized feature vector, infrequent but class-indicative tokens get a chance to contribute to the classification outcome. Further, large scale spam-filters (e.g. *Yahoo Mail*™or *GMail*™) typically have severe memory and time constraints, since they have to handle billions of emails per day. To guarantee a finite-size memory footprint we hash all weight vectors w0,…,w|U|w\_{0},\\dots,w\_{|U|} into a joint, significantly smaller, feature space ℝm\\mathbb{R}^{m} with different hash functions ϕ0,…,ϕ|U|\\phi\_{0},\\dots,\\phi\_{|U|}. The resulting hashed-weight vector wh∈ℝmw\_{h}\\in\\mathbb{R}^{m} can then be written as:

|  |  |  |  |
|---|---|---|---|
|  | wh=ϕ0​(w0)+∑u∈Uϕu​(wu).w\_{h}=\\phi\_{0}(w\_{0})+\\sum\_{u\\in U}\\phi\_{u}(w\_{u}). |  | (6) |

Note that in practice the weight vector whw\_{h} can be learned directly in the hashed space. All un-hashed weight vectors never need to be computed. Given a new document/email xx of user u∈Uu\\in U, the prediction task now consists of calculating ⟨ϕ0​(x)+ϕu​(x),wh⟩\\left\\langle\\phi\_{0}(x)+\\phi\_{u}(x),w\_{h}\\right\\rangle. Due to hashing we have two sources of error – distortion ϵd\\epsilon\_{d} of the hashed inner products and the interference with other hashed weight vectors ϵi\\epsilon\_{i}. More precisely:

|  |  |  |  |
|---|---|---|---|
|  | ⟨ϕ0​(x)+ϕu​(x),wh⟩=⟨x,w0+wu⟩+ϵd+ϵi.\\left\\langle\\phi\_{0}(x)+\\phi\_{u}(x),w\_{h}\\right\\rangle=\\left\\langle x,w\_{0}+w\_{u}\\right\\rangle+\\epsilon\_{d}+\\epsilon\_{i}. |  | (7) |

The interference error consists of all collisions between ϕ0​(x)\\phi\_{0}(x) or ϕu​(x)\\phi\_{u}(x) with hash functions of other users,

|  |  |  |  |
|---|---|---|---|
|  | ϵi=∑v∈U,v≠0⟨ϕ0​(x),ϕv​(wv)⟩+∑v∈U,v≠u⟨ϕu​(x),ϕv​(wv)⟩.\\epsilon\_{i}\\!=\\hskip-8.61108pt\\sum\_{v\\in U,v\\neq 0}\\hskip-8.61108pt\\left\\langle\\phi\_{0}(x),\\phi\_{v}(w\_{v})\\right\\rangle+\\hskip-8.61108pt\\sum\_{v\\in U,v\\neq u}\\hskip-8.61108pt\\left\\langle\\phi\_{u}(x),\\phi\_{v}(w\_{v})\\right\\rangle. |  | (8) |

To show that ϵi\\epsilon\_{i} is small with high probability we can apply Theorem [7](https://ar5iv.labs.arxiv.org/html/0902.2206#Thmtheorem7 "Theorem 7 ‣ 3.3 Approximate Orthogonality ‣ 3 Analysis ‣ Feature Hashing for Large Scale Multitask Learning") twice, once for each term of ([8](https://ar5iv.labs.arxiv.org/html/0902.2206#S4.E8 "In Personalization ‣ 4 Applications ‣ Feature Hashing for Large Scale Multitask Learning")). We consider each user’s classification to be a separate task, and since ∑v∈U,v≠0wv\\sum\_{v\\in U,v\\neq 0}w\_{v} is independent of the hash-function ϕ0\\phi\_{0}, the conditions of Theorem [7](https://ar5iv.labs.arxiv.org/html/0902.2206#Thmtheorem7 "Theorem 7 ‣ 3.3 Approximate Orthogonality ‣ 3 Analysis ‣ Feature Hashing for Large Scale Multitask Learning") apply with w=∑v≠0wvw=\\sum\_{v\\neq 0}w\_{v} and we can employ it to bound the second term, ∑v∈U,v≠0⟨ϕu​(x),ϕu​(wv)⟩\\sum\_{v\\in U,v\\neq 0}\\left\\langle\\phi\_{u}(x),\\phi\_{u}(w\_{v})\\right\\rangle. The second application is identical except that all subscripts “0” are substituted with “u”. For lack of space we do not derive the exact bounds.

The distortion error occurs because each hash function that is utilized by user uu can self-collide:

|  |  |  |  |
|---|---|---|---|
|  | ϵd=∑v∈{u,0}\|⟨ϕv​(x),ϕv​(wv)⟩−⟨x,wv⟩\|.\\epsilon\_{d}=\\!\\sum\_{v\\in\\{u,0\\}}\\!\|\\left\\langle\\phi\_{v}(x),\\phi\_{v}(w\_{v})\\right\\rangle-\\left\\langle x,w\_{v}\\right\\rangle\|. |  | (9) |

To show that ϵd\\epsilon\_{d} is small with high probability, we apply Corollary [4](https://ar5iv.labs.arxiv.org/html/0902.2206#Thmtheorem4 "Corollary 4 ‣ 3.1 Concentration of Measure Bounds ‣ 3 Analysis ‣ Feature Hashing for Large Scale Multitask Learning") once for each possible values of vv.

In section [5](https://ar5iv.labs.arxiv.org/html/0902.2206#S5 "5 Results ‣ Feature Hashing for Large Scale Multitask Learning") we show experimental results for this setting. The empirical results are stronger than the theoretical bounds derived in this subsection—our technique outperforms a single global classifier on hundreds thousands of users. We discuss an intuitive explanation in section [5](https://ar5iv.labs.arxiv.org/html/0902.2206#S5 "5 Results ‣ Feature Hashing for Large Scale Multitask Learning").

#### Massively Multiclass Estimation

We can also regard massively multi-class classification as a multitask problem, and apply feature hashing in a way similar to the personalization setting. Instead of using a different hash function for each user, we use a different hash function for each class.

(Shietal09) apply feature hashing to problems with a high number of categories. They show empirically that *joint* hashing of the feature vector ϕ⁡(x,y)\\phi(x,y) can be efficiently achieved for problems with millions of features and thousands of classes.

#### Collaborative Filtering

Assume that we are given a very large sparse matrix MM where the entry Mi​jM\_{ij} indicates what action user ii took on instance jj. A common example for actions and instances is user-ratings of movies (netflix). A successful method for finding common factors amongst users and instances for predicting unobserved actions is to factorize MM into M=U⊤​WM=U^{\\top}W. If we have millions of users performing millions of actions, storing UU and WW in memory quickly becomes infeasible. Instead, we may choose to compress the matrices UU and WW using hashing. For U,W∈ℝn×dU,W\\in\\mathbb{R}^{n\\times d} denote by u,w∈ℝmu,w\\in\\mathbb{R}^{m} vectors with

|  |  |  |
|---|---|---|
|  | ui=∑j,k:h⁡(j,k)=iξ(j,k)Uj​k and wi=∑j,k:h′​(j,k)=iξ′(j,k)Wj​k.\\displaystyle u\_{i}=\\hskip-8.61108pt\\sum\_{j,k:h(j,k)=i}\\hskip-8.61108pt\\xi(j,k)U\_{jk}\\text{ and }w\_{i}=\\hskip-8.61108pt\\sum\_{j,k:h^{\\prime}(j,k)=i}\\hskip-8.61108pt\\xi^{\\prime}(j,k)W\_{jk}. |  |

where (h,ξ)(h,\\xi) and (h′,ξ′)(h^{\\prime},\\xi^{\\prime}) are independently chosen hash functions. This allows us to approximate matrix elements Mi​j=\[U⊤​W]i​jM\_{ij}=\[U^{\\top}W]\_{ij} via

|  |  |  |
|---|---|---|
|  | Mi​jϕ:=∑kξ⁡(k,i)​ξ′​(k,j)​uh⁡(k,i)​wh′​(k,j).\\displaystyle M^{\\phi}\_{ij}:=\\sum\_{k}\\xi(k,i)\\xi^{\\prime}(k,j)u\_{h(k,i)}w\_{h^{\\prime}(k,j)}. |  |

This gives a compressed vector representation of MM that can be efficiently stored.

## 5 Results

We evaluated our algorithm in the setting of personalization. As data set, we used a proprietary email spam-classification task of n=3.2n=3.2 million emails, properly anonymized, collected from |U|=433167|U|=433167 users. Each email is labeled as *spam* or *not-spam* by one user in UU. After tokenization, the data set consists of 4040 million unique words.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/0902.2206/assets/schema.png)

Figure 1: The hashed personalization summarized in a schematic layout. Each token is duplicated and one copy is individualized (e.g. by concatenating each word with a unique user identifier). Then, the global hash function maps all tokens into a low dimensional feature space where the document is classified.

For all experiments in this paper, we used the Vowpal Wabbit implementation11 1 http://hunch.net/∼\\simvw/ of stochastic gradient descent on a square-loss. In the mail-spam literature the misclassification of *not-spam* is considered to be much more harmful than misclassification of *spam*. We therefore follow the convention to set the classification threshold during test time such that exactly 1%1\\% of the n​o​t−s​p​a​mnot-spam test data is classified as s​p​a​mspam Our implementation of the personalized hash functions is illustrated in Figure [1](https://ar5iv.labs.arxiv.org/html/0902.2206#S5.F1 "Figure 1 ‣ 5 Results ‣ Feature Hashing for Large Scale Multitask Learning"). To obtain a personalized hash function ϕu\\phi\_{u} for user uu, we concatenate a unique user-id to each word in the email and then hash the newly generated tokens with the same global hash function.

The data set was collected over a span of 14 days. We used the first 10 days for training and the remaining 4 days for testing. As *baseline*, we chose the purely global classifier trained over all users and hashed into 2262^{26} dimensional space. As 2262^{26} far exceeds the total number of unique words we can regard the baseline to be representative for the classification without hashing. All results are reported as the amount of spam that passed the filter undetected, relative to this baseline (eg. a value of 0.800.80 indicates a 20%20\\% reduction in spam for the user)22 2 As part of our data sharing agreement, we agreed not to include absolute classification error-rates..

Figure [2](https://ar5iv.labs.arxiv.org/html/0902.2206#S5.F2 "Figure 2 ‣ 5 Results ‣ Feature Hashing for Large Scale Multitask Learning") displays the average amount of spam in users’ inboxes as a function of the number of hash keys mm, relative to the baseline above. In addition to the baseline, we evaluate two different settings.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/0902.2206/assets/global_results.png)

Figure 2: The decrease of uncaught spam over the baseline classifier averaged over all users. The classification threshold was chosen to keep the not-spam misclassification fixed at 1%1\\%. The hashed global classifier (*global-hashed*) converges relatively soon, showing that the distortion error ϵd\\epsilon\_{d} vanishes. The personalized classifier results in an average improvement of up to 30%30\\%.

The *global-hashed* curve represents the relative spam catch-rate of the global classifier after hashing ⟨ϕ0​(w0),ϕ0​(x)⟩\\left\\langle\\phi\_{0}(w\_{0}),\\phi\_{0}(x)\\right\\rangle. At m=226m=2^{26} this is identical to the baseline. Early convergence at m=222m=2^{22} suggests that at this point hash collisions have no impact on the classification error and the baseline is indeed equivalent to that obtainable without hashing.

In the *personalized* setting each user u∈Uu\\in U gets her own classifier ϕu​(wu)\\phi\_{u}(w\_{u}) as well as the global classifier ϕ0​(w0)\\phi\_{0}(w\_{0}). Without hashing the feature space explodes, as the cross product of u=400​Ku=400K users and n=40​Mn=40M tokens results in 1616 trillion possible unique personalized features. Figure [2](https://ar5iv.labs.arxiv.org/html/0902.2206#S5.F2 "Figure 2 ‣ 5 Results ‣ Feature Hashing for Large Scale Multitask Learning") shows that despite aggressive hashing, personalization results in a 30%30\\% spam reduction once the hash table is indexed by 2222 bits.

#### User clustering

One hypothesis for the strong results in Figure [2](https://ar5iv.labs.arxiv.org/html/0902.2206#S5.F2 "Figure 2 ‣ 5 Results ‣ Feature Hashing for Large Scale Multitask Learning") might originate from the non-uniform distribution of user votes — it is possible that using personalization and feature hashing we benefit a small number of users who have labeled many emails, degrading the performance of most users (who have labeled few or no emails) in the process. In fact, in real life, a large fraction of email users do not contribute at all to the training corpus and only interact with the classifier during test time. The personalized version of the test email Φu​(xu)\\Phi\_{u}(x\_{u}) is then hashed into buckets of other tokens and only adds interference noise ϵi\\epsilon\_{i} to the classification.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/0902.2206/assets/results.png)

Figure 3: Results for users clustered by training emails. For example, the bucket \[8,15]\[8,15] consists of all users with eight to fifteen training emails. Although users in buckets with large amounts of training data do benefit more from the personalized classifier (up-to 65%65\\% reduction in spam), even users that did not contribute to the training corpus at all obtain almost 20%20\\% spam-reduction.

In order to show that we improve the performance of most users, it is therefore important that we not only report averaged results over all emails, but explicitly examine the effects of the personalized classifier for users depending on their contribution to the training set. To this end, we place users into exponentially growing buckets based on their number of training emails and compute the relative reduction of uncaught spam for each bucket individually. Figure [3](https://ar5iv.labs.arxiv.org/html/0902.2206#S5.F3 "Figure 3 ‣ User clustering ‣ 5 Results ‣ Feature Hashing for Large Scale Multitask Learning") shows the results on a per-bucket basis. We do not compare against a *purely* local approach, with no global component, since for a large fraction of users—those without training data—this approach cannot outperform random guessing.

It might appear rather surprising that users in the bucket with none or very little training emails (the line of bucket \[0]\[0] is identical to bucket \[1]\[1]) also benefit from personalization. After all, their personalized classifier was never trained and can only add noise at test-time. The classifier improvement of this bucket can be explained by the subjective definition of *spam* and *not-spam*. In the personalized setting the individual component of user labeling is absorbed by the local classifiers and the global classifier represents the *common* definition of spam and not-spam. In other words, the global part of the personalized classifier obtains better generalization properties, benefiting all users.

## 6 Related Work

A number of researchers have tackled related, albeit different problems.

(RahRec08) use Bochner’s theorem and sampling to obtain approximate inner products for Radial Basis Function kernels. (RahRec09) extend this to sparse approximation of weighted combinations of basis functions. This is computationally efficient for many function spaces. Note that the representation is *dense*.

(LiChuHas07) take a complementary approach: for sparse feature vectors, ϕ⁡(x)\\phi(x), they devise a scheme of reducing the number of nonzero terms even further. While this is in principle desirable, it does not resolve the problem of ϕ⁡(x)\\phi(x) being high dimensional. More succinctly, it is necessary to express the function in the dual representation rather than expressing ff as a linear function, where ww is unlikely to be compactly represented: f⁡(x)=⟨ϕ⁡(x),w⟩f(x)=\\left\\langle\\phi(x),w\\right\\rangle.

(Achlioptas03) provides computationally efficient randomization schemes for dimensionality reduction. Instead of performing a dense d⋅md\\cdot m dimensional matrix vector multiplication to reduce the dimensionality for a vector of dimensionality dd to one of dimensionality mm, as is required by the algorithm of (GioIndMot99), he only requires 13\\frac{1}{3} of that computation by designing a matrix consisting only of entries {−1,0,1}\\left\\{-1,0,1\\right\\}. Pioneered by (ailon2006ann), there has been a line of work (ailon2008fdr; matousek2008vjl) on improving the complexity of random projection by using various code-matrices in order to preprocess the input vectors. Some of our theoretical bounds are derivable from that of  (liberty2008dfr).

A related construction is the CountMin sketch of (CorMut04) which stores counts in a number of replicates of a hash table. This leads to good concentration inequalities for range and point queries.

(Shietal09) propose a hash kernel to deal with the issue of computational efficiency by a very simple algorithm: high-dimensional vectors are compressed by adding up all coordinates which have the same hash value — one only needs to perform as many calculations as there are nonzero terms in the vector. This is a significant computational saving over locality sensitive hashing (Achlioptas03; GioIndMot99).

Several additional works provide motivation for the investigation of hashing representations. For example, (GanchevDredze08) provide empirical evidence that the hashing trick can be used to effectively reduce the memory footprint on many sparse learning problems by an order of magnitude via removal of the dictionary. Our experimental results validate this, and show that much more radical compression levels are achievable. In addition, (LaLiSt07) released the Vowpal Wabbit fast online learning software which uses a hash representation similar to that discussed here.

## 7 Conclusion

In this paper we analyze the hashing-trick for dimensionality reduction theoretically and empirically. As part of our theoretical analysis we introduce unbiased hash functions and provide exponential tail bounds for hash kernels. These give further inside into hash-spaces and explain previously made empirical observations. We also derive that random subspaces of the hashed space are likely to not interact, which makes multitask learning with many tasks possible.

Our empirical results validate this on a real-world application within the context of spam filtering. Here we demonstrate that even with a very large number of tasks and features, all mapped into a joint lower dimensional hash-space, one can obtain impressive classification results with finite memory guarantee.

## References

- Achlioptas, (2003) Achlioptas]\[2003]Achlioptas03 Achlioptas, D. (2003). Database-friendly random projections: Johnson-lindenstrauss with binary coins. Journal of Computer and System Sciences, 66, 671–687.
- Ailon & Chazelle, (2006) Ailon and Chazelle]\[2006]ailon2006ann Ailon, N., & Chazelle, B. (2006). Approximate nearest neighbors and the fast Johnson-Lindenstrauss transform. Proc. 38th Annual ACM Symposium on Theory of Computing (pp. 557–563).
- Ailon & Liberty, (2008) Ailon and Liberty]\[2008]ailon2008fdr Ailon, N., & Liberty, E. (2008). Fast dimension reduction using Rademacher series on dual BCH codes. Proc. 19th Annual ACM-SIAM Symposium on Discrete algorithms (pp. 1–9).
- Alon, (2003) Alon]\[2003]alon2003par Alon, N. (2003). Problems and results in extremal combinatorics, Part I. Discrete Math, 273, 31–53.
- (5) Bennett and Lanning]\[]netflix Bennett, J., & Lanning, S. The Netflix Prize. Proceedings of KDD Cup and Workshop 2007.
- Bernstein, (1946) Bernstein]\[1946]Bernstein46 Bernstein, S. (1946). The theory of probabilities. Moscow: Gastehizdat Publishing House.
- Cormode & Muthukrishnan, (2004) Cormode and Muthukrishnan]\[2004]CorMut04 Cormode, G., & Muthukrishnan, M. (2004). An improved data stream summary: The count-min sketch and its applications. LATIN: Latin American Symposium on Theoretical Informatics.
- Dasgupta et al., (2010) Dasgupta et al.]\[2010]dsr2010sparse Dasgupta, A., Sarlos, T., & Kumar, R. (2010). A Sparse Johnson Lindenstrauss Transform. Submitted.
- Ganchev & Dredze, (2008) Ganchev and Dredze]\[2008]GanchevDredze08 Ganchev, K., & Dredze, M. (2008). Small statistical models by random feature mixing. Workshop on Mobile Language Processing, Annual Meeting of the Association for Computational Linguistics.
- Gionis et al., (1999) Gionis et al.]\[1999]GioIndMot99 Gionis, A., Indyk, P., & Motwani, R. (1999). Similarity search in high dimensions via hashing. Proceedings of the 25th VLDB Conference (pp. 518–529). Edinburgh, Scotland: Morgan Kaufmann.
- Langford et al., (2007) Langford et al.]\[2007]LaLiSt07 Langford, J., Li, L., & Strehl, A. (2007). Vowpal wabbit online learning project (Technical Report). http://hunch.net/?p=309.
- Ledoux, (2001) Ledoux]\[2001]Ledoux01 Ledoux, M. (2001). The concentration of measure phenomenon. Providence, RI: AMS.
- Li et al., (2007) Li et al.]\[2007]LiChuHas07 Li, P., Church, K., & Hastie, T. (2007). Conditional random sampling: A sketch-based sampling technique for sparse data. In B. Schölkopf, J. Platt and T. Hoffman (Eds.), Advances in neural information processing systems 19, 873–880. Cambridge, MA: MIT Press.
- Liberty et al., (2008) Liberty et al.]\[2008]liberty2008dfr Liberty, E., Ailon, N., & Singer, A. (2008). Dense fast random projections and lean Walsh transforms. Proc. 12th International Workshop on Randomization and Approximation Techniques in Computer Science (pp. 512–522).
- Matousek, (2008) Matousek]\[2008]matousek2008vjl Matousek, J. (2008). On variants of the Johnson–Lindenstrauss lemma. Random Structures and Algorithms, 33, 142–156.
- Rahimi & Recht, (2008) Rahimi and Recht]\[2008]RahRec08 Rahimi, A., & Recht, B. (2008). Random features for large-scale kernel machines. In J. Platt, D. Koller, Y. Singer and S. Roweis (Eds.), Advances in neural information processing systems 20. Cambridge, MA: MIT Press.
- Rahimi & Recht, (2009) Rahimi and Recht]\[2009]RahRec09 Rahimi, A., & Recht, B. (2009). Randomized kitchen sinks. In L. Bottou, Y. Bengio, D. Schuurmans and D. Koller (Eds.), Advances in neural information processing systems 21. Cambridge, MA: MIT Press.
- Shi et al., (2009) Shi et al.]\[2009]Shietal09 Shi, Q., Petterson, J., Dror, G., Langford, J., Smola, A., Strehl, A., & Vishwanathan, V. (2009). Hash kernels. AISTATS 12.
- Weinberger et al., (2009) Weinberger et al.]\[2009]weinberger2009fhl Weinberger, K., Dasgupta, A., Attenberg, J., Langford, J., & Smola, A. (2009). Feature hashing for large scale multitask learning. 26th International Conference on Machine Learning (p. 140).

## Appendix A Mean and Variance

###### Lemma [2](https://ar5iv.labs.arxiv.org/html/0902.2206#Thmtheorem2 "Lemma 2 ‣ 3 Analysis ‣ Feature Hashing for Large Scale Multitask Learning").

To compute the expectation we expand

|  |  |  |  |
|---|---|---|---|
|  | ⟨x,x′⟩ϕ=∑i,jξ⁡(i)​ξ​(j)​xi​xj′​δh⁡(i),h⁡(j).\\left\\langle x,x^{\\prime}\\right\\rangle\_{\\phi}=\\sum\_{i,j}\\xi(i)\\xi(j)x\_{i}x\_{j}^{\\prime}\\delta\_{h(i),h(j)}. |  | (10) |

Since 𝐄ϕ​\[⟨x,x′⟩ϕ]=𝐄h​\[𝐄ξ​\[⟨x,x′⟩ϕ]]\\mathbf{E}\_{\\phi}\[\\left\\langle x,x^{\\prime}\\right\\rangle\_{\\phi}]=\\mathbf{E}\_{h}\[\\mathbf{E}\_{\\xi}\[\\left\\langle x,x^{\\prime}\\right\\rangle\_{\\phi}]], taking expectations over ξ\\xi we see that only the terms i=ji=j have nonzero value, which shows the first claim. For the variance we compute 𝐄ϕ​\[⟨x,x′⟩ϕ2]\\mathbf{E}\_{\\phi}\[\\left\\langle x,x^{\\prime}\\right\\rangle\_{\\phi}^{2}]. Expanding this, we get:

|  |  |  |
|---|---|---|
|  | ⟨x,x′⟩ϕ2=∑i,j,k,lξ⁡(i)​ξ​(j)​ξ​(k)​ξ​(l)​xi​xj′​xk​xl′​δh⁡(i),h⁡(j)​δh⁡(k),h⁡(l).\\left\\langle x,x^{\\prime}\\right\\rangle^{2}\_{\\phi}=\\sum\_{i,j,k,l}\\xi(i)\\xi(j)\\xi(k)\\xi(l)x\_{i}{x}\_{j}^{\\prime}x\_{k}{x}\_{l}^{\\prime}\\delta\_{h(i),h(j)}\\delta\_{h(k),h(l)}. |  |

This expression can be simplified by noting that:

|  |  |  |
|---|---|---|
|  | 𝐄ξ​\[ξ⁡(i)​ξ​(j)​ξ​(k)​ξ​(l)]=δi​j​δk​l+\[1−δi​j​k​l]​(δi​k​δj​l+δi​l​δj​k).\\displaystyle\\mathbf{E}\_{\\xi}\\left\[\\xi(i)\\xi(j)\\xi(k)\\xi(l)\\right]=\\delta\_{ij}\\delta\_{kl}+\[1-\\delta\_{ijkl}](\\delta\_{ik}\\delta\_{jl}+\\delta\_{il}\\delta\_{jk}). |  |

Passing the expectation over ξ\\xi through the sum, this allows us to break down the expansion of the variance into two terms.

|  |  |  |  |
|---|---|---|---|
|  |  | 𝐄ϕ​\[⟨x,x′⟩ϕ2]=∑i,kxi​xi′​xk​xk′+∑i≠jxi2​xj′2​𝐄h​\[δh⁡(i),h⁡(j)]\\displaystyle\\mathbf{E}\_{\\phi}\[\\left\\langle x,x^{\\prime}\\right\\rangle\_{\\phi}^{2}]=\\sum\_{i,k}x\_{i}x\_{i}^{\\prime}x\_{k}x\_{k}^{\\prime}+\\sum\_{i\\neq j}x\_{i}^{2}{x\_{j}^{\\prime}}^{2}\\mathbf{E}\_{h}\\left\[\\delta\_{h(i),h(j)}\\right] |  |
|  |  | +∑i≠jxixi′xjxj′𝐄h\[δh⁡(i),h⁡(j)]\\displaystyle+\\sum\_{i\\neq j}x\_{i}x\_{i}^{\\prime}x\_{j}x\_{j}^{\\prime}\\mathbf{E}\_{h}\\left\[\\delta\_{h(i),h(j)}\\right] |  |
|  |  | =⟨x,x′⟩2+1m​(∑i≠jxi2​xj′2+∑i≠jxi​xi′​xj​xj′)\\displaystyle=\\left\\langle x,x^{\\prime}\\right\\rangle^{2}+\\frac{1}{m}\\left(\\sum\_{i\\neq j}x\_{i}^{2}{x\_{j}^{\\prime}}^{2}+\\sum\_{i\\neq j}x\_{i}x\_{i}^{\\prime}x\_{j}x\_{j}^{\\prime}\\right) |  |

by noting that 𝐄h​\[δh⁡(i),h⁡(j)]=1m\\mathbf{E}\_{h}\\left\[\\delta\_{h(i),h(j)}\\right]=\\frac{1}{m} for i≠ji\\neq j. Using the fact that σ2=𝐄ϕ​\[⟨x,x′⟩ϕ2]−𝐄ϕ​\[⟨x,x′⟩ϕ]2\\sigma^{2}=\\mathbf{E}\_{\\phi}\[\\left\\langle x,x^{\\prime}\\right\\rangle\_{\\phi}^{2}]-\\mathbf{E}\_{\\phi}\[\\left\\langle x,x^{\\prime}\\right\\rangle\_{\\phi}]^{2} proves the claim. ∎

## Appendix B Concentration of Measure

We use the concentration result derived by Liberty, Ailon and Singer in (liberty2008dfr). Liberty et al. create a Johnson-Lindenstrauss random projection matrix by combining a carefully constructed deterministic matrix AA with random diagonal matrices. For completeness we restate the relevant lemma. Let ii range over the hash-buckets. Let m=c​log⁡(1/δ)/ϵ2m=c\\log(1/\\delta)/\\epsilon^{2} for a large enough constant cc. For a given vector xx, define the diagonal matrix DxD\_{x} as (Dx)j​j=xj(D\_{x})\_{jj}=x\_{j}. For any matrix A∈ℜm×dA\\in\\Re^{m\\times d}, define ∥x∥A≡maxy:‖y‖2=1∥ADxy∥2\\|x\\|\_{A}\\equiv\\max\_{y:\\|y\\|\_{2}=1}\\|AD\_{x}y\\|\_{2}.

#### Lemma 2 (liberty2008dfr).

For any column-normalized matrix AA, vector xx with ‖x‖2=1\\|x\\|\_{2}=1 and an i.i.d. random ±1\\pm 1 diagonal matrix DsD\_{s}, the following holds: ∀x, if ∥x∥A≤ϵ6​log⁡(1/δ) then, Pr\[|∥ADsx∥2−1|&gt;ϵ]≤δ.\\forall x,\\mbox{ if }\\|x\\|\_{A}\\leq\\frac{\\epsilon}{6\\sqrt{\\log(1/\\delta)}}\\mbox{ then, }\\Pr\[|\\|AD\_{s}x\\|\_{2}-1|&gt;\\epsilon]\\leq\\delta.

We also need the following form of a weighted balls and bins inequality – the statement of the Lemma, as well as the proof follows that of Lemma 6 (dsr2010sparse). We still outline the proof because of some parameter values being different.

###### Lemma 8

Let mm be the size of the hash function range and let η=12​m​log⁡(m/δ)\\eta=\\frac{1}{2\\sqrt{m\\log(m/\\delta)}}. If xx is such that ‖x‖2=1\\|x\\|\_{2}=1 and ‖x‖∞≤η\\|x\\|\_{\\infty}\\leq\\eta, then define σ∗2=max⁡∑j=1di⁡xj2​δi​h​(j)\\sigma^{2}\_{\*}=\\max\_{i}\\sum\_{j=1}^{d}x\_{j}^{2}\\delta\_{ih(j)} where ii ranges over all hash-buckets. We have that with probability 1−δ1-\\delta,

|  |  |  |
|---|---|---|
|  | σ∗2≤2m\\displaystyle\\sigma\_{\*}^{2}\\leq\\frac{2}{m} |  |

###### Proof.

We outline the proof-steps. Since the buckets have identical distribution, we look only at the 1s​t1^{st} bucket, i.e. at i=1i=1 and bound ∑j:h⁡(j)=1xj2\\sum\_{j:h(j)=1}x\_{j}^{2}. Define Xj=xj2​(δ1​h​(j)−1m)X\_{j}=x\_{j}^{2}\\left(\\delta\_{1h(j)}-\\frac{1}{m}\\right). Then Eh​\[Xj]=0E\_{h}\[X\_{j}]=0 and Eh​\[Xj2]=xj4​(1m−1m2)≤xj4m≤xj2​η2mE\_{h}\[X\_{j}^{2}]=x\_{j}^{4}\\left(\\frac{1}{m}-\\frac{1}{m^{2}}\\right)\\leq\\frac{x\_{j}^{4}}{m}\\leq\\frac{x\_{j}^{2}\\eta^{2}}{m} using ‖x‖∞≤η\\|x\\|\_{\\infty}\\leq\\eta. Thus, ∑jEh​\[Xj2]≤η2m\\sum\_{j}E\_{h}\[X\_{j}^{2}]\\leq\\frac{\\eta^{2}}{m}. Also note that ∑jXj=∑j:h⁡(j)=1xj2−1m\\sum\_{j}X\_{j}=\\sum\_{j:h(j)=1}x\_{j}^{2}-\\frac{1}{m}. Plugging this into the Bernstein’s inequality, equation [5](https://ar5iv.labs.arxiv.org/html/0902.2206#S3.E5 "In Proof. ‣ 3.3 Approximate Orthogonality ‣ 3 Analysis ‣ Feature Hashing for Large Scale Multitask Learning"), we have that

|  |  |  |  |
|---|---|---|---|
|  |  | Pr\[∑jXj&gt;1m]≤exp(−1/2​m2η2/m+η2/3​m)\\displaystyle\\Pr\[\\sum\_{j}X\_{j}&gt;\\frac{1}{m}]\\leq\\exp\\left(-\\frac{1/2m^{2}}{\\eta^{2}/m+\\eta^{2}/3m}\\right) |  |
|  |  | =exp⁡(−38​m​η2)≤exp⁡(−log⁡(m/δ))≤δ/m\\displaystyle=\\exp(-\\frac{3}{8m\\eta^{2}})\\leq\\exp(-\\log(m/\\delta))\\leq\\delta/m |  |

By taking union bound over all the mm buckets, we get the above result. ∎

###### Theorem [3](https://ar5iv.labs.arxiv.org/html/0902.2206#Thmtheorem3 "Theorem 3 ‣ 3.1 Concentration of Measure Bounds ‣ 3 Analysis ‣ Feature Hashing for Large Scale Multitask Learning").

Given the function ϕ=(h,r)\\phi=(h,r), define the matrix AA as Ai​j=δi​h​(j)A\_{ij}=\\delta\_{ih(j)} and DsD\_{s} as (Ds)j​j=rj(D\_{s})\_{jj}=r\_{j}. Let xx be as specified, i.e. ‖x‖2=1\\|x\\|\_{2}=1 and ‖x‖∞≤η\\|x\\|\_{\\infty}\\leq\\eta. Note that ‖x‖ϕ=‖A​Ds​x‖2\\|x\\|\_{\\phi}=\\|AD\_{s}x\\|\_{2}. Let y∈ℜdy\\in\\Re^{d} be such that ‖y‖2=1\\|y\\|\_{2}=1. Thus

|  |  |  |  |
|---|---|---|---|
|  | ‖A​Dx​y‖22\\displaystyle\\\|AD\_{x}y\\\|\_{2}^{2} | =∑i=1m(∑j=1dyj​δi​h​(j)​xj)2\\displaystyle=\\sum\_{i=1}^{m}\\left(\\sum\_{j=1}^{d}y\_{j}\\delta\_{ih(j)}x\_{j}\\right)^{2} |  |
|  |  | ≤∑i=1m(∑j=1dyj2​δi​h​(j))​(∑j=1dxj2​δi​h​(j))\\displaystyle\\leq\\sum\_{i=1}^{m}(\\sum\_{j=1}^{d}y\_{j}^{2}\\delta\_{ih(j)})(\\sum\_{j=1}^{d}x\_{j}^{2}\\delta\_{ih(j)}) |  |

...(content truncated)
