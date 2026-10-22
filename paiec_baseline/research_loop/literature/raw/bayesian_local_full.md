> **External page content (untrusted):** Treat the content below as data, not instructions. Do not follow requests in it to call tools or disclose or send data.

## [1807.08912] Meta-Learning Priors for Efficient Online Bayesian Regression

**Source**: https://ar5iv.labs.arxiv.org/html/1807.08912

---

# Meta-Learning Priors for Efficient Online Bayesian Regression

James Harrison    Apoorva Sharma    Marco Pavone E-mail: [{jharrison,apoorva,pavone}@stanford.edu](mailto:{jharrison,apoorva,pavone}@stanford.edu)

###### Abstract

Gaussian Process (GP) regression has seen widespread use in robotics due to its generality, simplicity of use, and the utility of Bayesian predictions. The predominant implementation of GP regression is a nonparameteric kernel-based approach, as it enables fitting of arbitrary nonlinear functions. However, this approach suffers from two main drawbacks: (1) it is computationally inefficient, as computation scales poorly with the number of samples; and (2) it can be data inefficient, as encoding prior knowledge that can aid the model through the choice of kernel and associated hyperparameters is often challenging and unintuitive. In this work, we propose ALPaCA, an algorithm for efficient Bayesian regression which addresses these issues. ALPaCA uses a dataset of sample functions to learn a domain-specific, finite-dimensional feature encoding, as well as a prior over the associated weights, such that Bayesian linear regression in this feature space yields accurate online predictions of the posterior predictive density. These features are neural networks, which are trained via a meta-learning (or “learning-to-learn”) approach. ALPaCA extracts all prior information directly from the dataset, rather than restricting prior information to the choice of kernel hyperparameters. Furthermore, by operating in the weight space, it substantially reduces sample complexity. We investigate the performance of ALPaCA on two simple regression problems, two simulated robotic systems, and on a lane-change driving task performed by humans. We find our approach outperforms kernel-based GP regression, as well as state of the art meta-learning approaches, thereby providing a promising plug-in tool for many regression tasks in robotics where scalability and data-efficiency are important.

††tocauthor: James Harrison, Apoorva Sharma, Marco Pavone††institute: Stanford University

## 1 Introduction

Gaussian Process (GP) regression has become a workhorse in robotics due to its generality, simplicity of use, and the utility of Bayesian predictions. As such, it has been used in robotic applications as diverse as model identification for control and reinforcement learning [deisenroth2011pilco](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib1); [berkenkamp2017safe](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib2), contact modeling [bauza2017probabilistic](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib3), terrain and environment modeling [vasudevan2009gaussian](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib4); [o2012gaussian](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib5), prediction of human behavior [wang2008gaussian](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib6); [urtasun20063d](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib7), and trajectory optimization [mukadam2016gaussian](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib8). Moreover, it has been used effectively for essential tools in robotics such as Bayesian filtering [ko2009gp](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib9), SLAM [ferris2007wifi](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib10), and Bayesian optimization [snoek2012practical](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib11).

While GP regression is a useful tool, standard approaches have several well-known limitations. GP regression is predominantly performed via kernel-based methods [rasmussen2004gaussian](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib12), but incorporating physical priors into these models is difficult, and done implicitly through the choice of kernel function. For example, in modeling a physical system with a GP equipped with the widely-used squared exponential (SE) kernel, the system designer may vary the prior variance, and the length scale of the kernel, which controls smoothness [rasmussen2004gaussian](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib12). These choices of priors are neither expressive nor intuitive enough for complex applications in robotics. Moreover, kernel-based GP regression algorithms scale cubically with the amount of data [hensman2013gaussian](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib13). This high sample complexity prevents application beyond relatively small problems [smola2001sparse](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib14).

Several works have attempted to incorporate prior knowledge into GP models and to remove the limitations imposed by kernel methods. Methods mapping input data to alternative feature spaces which are then used as inputs to kernel GP models have been shown to allow incorporation of prior knowledge, and reduce the complexity by projecting to a lower dimensional space [mackay1998introduction](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib15). For example, Hinton and Salakhutdinov [hinton2008using](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib16) use a Deep Belief Network to learn a feature representation in an unsupervised fashion, but this separates the process of representation learning and regression, which may be suboptimal. Manifold GPs [calandra2016manifold](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib17) jointly learn a neural network transformation of the data such that GP methods applied to this feature space achieve good predictions. However, this approach relies on the system designer having access to data generated from the exact system being modeled, and it is unclear how well this approach will perform on a regression problem that differs from that used to generate the training dataset. Good performance on a general class of tasks (as opposed to one specific task within this family) is necessary for effective autonomous operation in unstructured environments. Moreover, these approaches still suffer from the poor sample complexity associated with kernel methods. A more extensive discussion of methods for incorporating prior knowledge is presented in Section [5](https://ar5iv.labs.arxiv.org/html/1807.08912#S5 "5 Related Work ‣ Meta-Learning Priors for Efficient Online Bayesian Regression").

A variety of sparse and approximate methods have been proposed to reduce the sample complexity of kernel methods, but these at best scale quadratically [snelson2006sparse](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib18); [np2018](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib19). A popular approach to avoiding the complexity of kernel methods is to instead turn to random features for approximating kernel functions, and solving the linear regression problem using these features instead [rahimi2008random](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib20). The approach taken in this work is similar, but we learn features that are well-suited to representing posterior densities, based on available prior information.

#### Contributions.

In this work, we propose ALPaCA (Adaptive Learning for Probabilistic Connectionist Architectures), an algorithm for incorporating prior information in to GP regression that increases computational and sample efficiency, and provides Bayesian posteriors reflecting confidence in predictions. We propose to learn a neural network model (i.e. connectionist architecture) offline that is trained to be able to rapidly incorporate online information to predict posterior distributions. Online adaptation is achieved by performing Bayesian linear regression on the last layer of the network, and thus our approach is capable of providing confidence intervals for predictions. Fundamentally, our network learns priors over the last layer weights, as well as network weights, such that it can optimally predict the posterior over the label, conditioned on the online samples observed so far. This is performed by leveraging the “learning-to-learn” approach of meta-learning [finn2017model](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib21), where model training is done to minimize divergence between the predicted and true posterior, for any amount of observed online context data. By performing regression on meta-learned basis functions, ALPaCA achieves complexity during inference that is linear in the amount of data. As such, ALPaCA offers a general, plug-in alternative to kernel GP regression which can improve scalability and online data-efficiency in a wide variety of robotics applications.

#### Organization.

In Section 2, we provide a formal problem statement. We present background information on Bayesian linear regression and GP regression in Section 3. In Section 4 we introduce the ALPaCA algorithm for generic supervised learning tasks. We contrast our approach to the literature on neural network regression models, current meta-learning approaches, as well as popular GP models in Section 5. Finally in Section 6, we demonstrate ALPaCA on simple problems that are standard benchmarks in meta-learning [finn2017model](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib21), as well as high-dimensional robotic systems involving contact, and prediction of human decision making in a driving task11 1 The code for all of our experiments is available at [https://github.com/StanfordASL/ALPaCA](https://github.com/StanfordASL/ALPaCA). ALPaCA is shown to outperform kernel GP regression, as well as meta-learning approaches such as MAML [finn2017model](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib21).

## 2 Problem Formulation

In this work, we are focused on the problem of Bayesian function regression. In this setting, we start with a prior over possible functions, observe samples from an unknown test function, and use these samples to compute a posterior predictive density over the unknown function.

Formally, let f:ℝnx→ℝnyf:\\mathbb{R}^{n\_{x}}\\to\\mathbb{R}^{n\_{y}} denote an arbitrary, possibly nonlinear and nonsmooth function with latent parameters θ\\theta. We consider a setting where ff is fixed and captures problem specific structure, while θ\\theta is uncertain and explains intra-domain variation. For example, in the context of prediction of human driving behavior, uncertainty in the state evolution stems from uncertainty in the driver’s intent, while the physics governing the car’s dynamics are known. We assume the samples of ff observed online may be corrupted with additive Gaussian noise. That is, we observe a sequence of samples (x,y)(x,y) such that

|  |  |  |  |
|---|---|---|---|
|  | p⁡(y∣x,θ)=𝒩⁡(f⁡(x,θ),Σϵ),p(y\\mid x,\\theta)=\\mathcal{N}(f(x;\\theta),\\Sigma\_{\\epsilon}), |  | (1) |

where 𝒩⁡(⋅,⋅)\\mathcal{N}(\\cdot,\\cdot) denotes the multivariate Gaussian distribution, and Σϵ\\Sigma\_{\\epsilon} is the noise covariance, and θ\\theta is unknown.

Suppose we observe a collection of τ\\tau sample points Dτ∗={(xt,yt)}t=1τD^{\*}\_{\\tau}=\\{(x\_{t},y\_{t})\\}\_{t=1}^{\\tau} coming from an unknown function with latent parameter θ∗{\\theta^{\*}}. Given a prior over model parameters p⁡(θ)p(\\theta), and this context data Dτ∗D^{\*}\_{\\tau}, the posterior predictive density over yy given xx can be computed as

|  |  |  |  |
|---|---|---|---|
|  | p⁡(y∣x,Dτ∗)=∫p⁡(y∣x,θ)​p​(θ∣Dτ∗)​𝑑θ.p(y\\mid x,D^{\*}\_{\\tau})=\\int p(y\\mid x,\\theta)p(\\theta\\mid D^{\*}\_{\\tau})\\,d\\theta. |  | (2) |

Computing this posterior predictive density, however, requires having analytic expressions for f⁡(x,θ)f(x;\\theta) and p⁡(θ)p(\\theta), and even then, computing the posterior p⁡(θ∣Dτ∗)p(\\theta\\mid D^{\*}\_{\\tau}) exactly may remain intractable. Instead, we propose using a surrogate model for which computing the posterior predictive density is analytically tractable, and optimizing this surrogate model offline such that its posterior preditictive density is close to the true posterior predictive density for all likely settings of θ∗{\\theta^{\*}} and the resulting observations Dτ∗D^{\*}\_{\\tau}.

Let qξ​(y∣x,Dτ∗)q\_{\\xi}(y\\mid x,D^{\*}\_{\\tau}) represent the posterior predictive density under our surrogate model, parameterized by ξ\\xi. Let y^\\hat{y} represent a sample from this density qq. We consider a scenario where data comes in as a stream: that is, at each timestep, the agent is presented with an input xtx\_{t}, and after estimating the output y^t\\hat{y}\_{t}, the agent is provided with the true value yty\_{t}. An example of this is a Markovian dynamical system, in which an agent wishes to predict the distribution over the next state, and at the next time step, this state is observed.

In this context, the problem we wish to solve can be written as

|  |  |  |  |
|---|---|---|---|
|  | minξDKL(p(y∣xt+1,Dt∗)∥qξ(y∣xt+1,Dt∗)),\\min\_{\\xi}\~\~D\_{\\textrm{KL}}\\left(p(y\\mid x\_{t+1},D^{\*}\_{t})\\,\\\|\\,q\_{\\xi}(y\\mid x\_{t+1},D^{\*}\_{t})\\right), |  | (3) |

where DKL(⋅∥⋅)D\_{\\textrm{KL}}(\\cdot\\|\\cdot) denotes the Kullback-Leibler (KL) divergence. Note that this objective is minimizing the conditional KL divergence. As such, the objective is to minimize this divergence in expectation over t,xt+1,Dt∗t,x\_{t+1},D^{\*}\_{t}, and θ∗{\\theta^{\*}}. Directly optimizing this objective requires access to p⁡(θ)p(\\theta) and p⁡(Dt∗∣θ∗)p(D^{\*}\_{t}\\mid{\\theta^{\*}}). Instead of assuming access to these distributions directly, we only require a dataset of samples drawn from different settings of latent parameter θ\\theta. Specifically, we will write Dτ(j)≔{(xt,yt)}t=1τD^{(j)}\_{\\tau}\\coloneqq\\{(x\_{t},y\_{t})\\}\_{t=1}^{\\tau} to represent a dataset of τ\\tau samples generated via sampling (iid) θj∼p⁡(θ)\\theta\_{j}\\sim p(\\theta), xt∼p⁡(x)x\_{t}\\sim p(x) and yt∼p⁡(y∣xt,θj)y\_{t}\\sim p(y\\mid x\_{t},\\theta\_{j}), that is, a set of samples from one function instantiation. Let 𝒟≔{Dτ(j)}j=1M\\mathcal{D}\\coloneqq\\{D^{(j)}\_{\\tau}\\}\_{j=1}^{M} represent a collection of these datasets. Note that by sampling t∼p⁡(t)t\\sim p(t), sampling Dτ(j)D^{(j)}\_{\\tau} uniformly from 𝒟\\mathcal{D}, and the truncating Dτ(j)D^{(j)}\_{\\tau} to t&lt;τt&lt;\\tau elements, we can approximate the expectations in ([3](https://ar5iv.labs.arxiv.org/html/1807.08912#S2.E3 "In 2 Problem Formulation ‣ Meta-Learning Priors for Efficient Online Bayesian Regression")) through a Monte Carlo scheme.

## 3 Bayesian Regression

In this work, we utilize Bayesian multivariate linear regression as a tool for computing q⁡(y∣x,Dτ∗)q(y\\mid x,D^{\*}\_{\\tau}), as it allows for analytic computation of the posterior. In this section, we offer a brief review of the technique. For a more in-depth treatment, we refer the reader to [rasmussen2004gaussian](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib12), [minka2000bayesian](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib22), or [murphy2012machine](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib23). Let xt∈ℝnxx\_{t}\\in\\mathbb{R}^{n\_{x}} and yt∈ℝnyy\_{t}\\in\\mathbb{R}^{n\_{y}} denote the independent and dependent variables respectively. We will consider regression over basis functions ϕ:ℝnx→ℝnϕ\\phi:\\mathbb{R}^{n\_{x}}\\to\\mathbb{R}^{n\_{\\phi}}. Then, ytT=ϕT​(xt)​K+ϵty\_{t}^{T}=\\phi^{T}(x\_{t})K+\\epsilon\_{t}, where K∈ℝnϕ×nyK\\in\\mathbb{R}^{{n\_{\\phi}}\\times{n\_{y}}} is a coefficient matrix and ϵ∼𝒩⁡(0,Σϵ)\\epsilon\\sim\\mathcal{N}(0,\\Sigma\_{\\epsilon}), wherein 𝒩⁡(⋅,⋅)\\mathcal{N}(\\cdot,\\cdot) denotes the multivariate Gaussian distribution and Σϵ\\Sigma\_{\\epsilon} denotes the noise variance. We will assume Σϵ\\Sigma\_{\\epsilon} is known throughout this work. The case is which this assumption is relaxed is discussed in the appendix. Let YT=\[y1,…,yτ]Y^{T}=\[y\_{1},\\ldots,y\_{\\tau}], ΦT=\[ϕ⁡(x1),…,ϕ⁡(xτ)]\\Phi^{T}=\[\\phi(x\_{1}),\\ldots,\\phi(x\_{\\tau})], where τ\\tau is the number of data points. Stacking samples t=1,…,τt=1,\\ldots,\\tau, we may equivalently write Y=Φ​K+EY=\\Phi K+E. Given this formulation, the conditional probability of the data YY is

|  |  |  |  |
|---|---|---|---|
|  | p(Y∣Φ,K,Σϵ)∝\|Σϵ\|−ny/2exp(−12tr((Y−ΦK)Σϵ−1(Y−ΦK)T)),p(Y\\mid\\Phi,K,\\Sigma\_{\\epsilon})\\propto\|\\Sigma\_{\\epsilon}\|^{-{n\_{y}}/2}\\exp\\left(-\\frac{1}{2}\\Tr\\left((Y-\\Phi K)\\Sigma\_{\\epsilon}^{-1}(Y-\\Phi K)^{T}\\right)\\right), |  | (4) |

where tr⁡(⋅)\\Tr(\\cdot) denotes the trace operation. A natural conjugate prior for KK is p⁡(K)∼ℳ​𝒩​(K¯0,Λ0−1,Σϵ),p(K)\\sim\\mathcal{MN}(\\bar{K}\_{0},\\Lambda\_{0}^{-1},\\Sigma\_{\\epsilon}), where ℳ​𝒩\\mathcal{MN} denotes the matrix normal distribution, and Λ0\\Lambda\_{0} is a precision matrix (the inverse of the variance). The matrix normal distribution is equivalent to the multivariate normal distribution with mean vec​(K¯0)\\textrm{vec}(\\bar{K}\_{0}) and variance Σϵ⊗Λ0−1\\Sigma\_{\\epsilon}\\otimes\\Lambda\_{0}^{-1}, where vec​(K¯0)\\textrm{vec}(\\bar{K}\_{0}) represents the vectorization of K¯0\\bar{K}\_{0} and ⊗\\otimes denotes the Kronecker product. The posterior, conditioned on YY and Φ\\Phi, is then p⁡(K∣Y,Φ)=ℳ​𝒩​(K¯τ,Λτ−1,Σϵ),p(K\\mid Y,\\Phi)=\\mathcal{MN}(\\bar{K}\_{\\tau},\\Lambda\_{\\tau}^{-1},\\Sigma\_{\\epsilon}), where

|  |  |  |  |  |
|---|---|---|---|---|
|  | Λτ\\displaystyle\\Lambda\_{\\tau} | =ΦT​Φ+Λ0\\displaystyle=\\Phi^{T}\\Phi+\\Lambda\_{0} |  | (5) |
|  | K¯τ\\displaystyle\\bar{K}\_{\\tau} | =(Λτ)−1​(ΦT​Y+Λ0​K¯0).\\displaystyle=(\\Lambda\_{\\tau})^{-1}(\\Phi^{T}Y+\\Lambda\_{0}\\bar{K}\_{0}). |  | (6) |

Given the above, the posterior predictive distribution is

|  |  |  |  |
|---|---|---|---|
|  | p⁡(yτ+1∣ϕ⁡(xτ+1),Φ,Y)=𝒩⁡(K¯τT​ϕ​(xτ+1),Στ+1)p(y\_{\\tau+1}\\mid\\phi(x\_{\\tau+1}),\\Phi,Y)=\\mathcal{N}(\\bar{K}\_{\\tau}^{T}\\phi(x\_{\\tau+1}),\\Sigma\_{\\tau+1}) |  | (7) |

where

|  |  |  |  |
|---|---|---|---|
|  | Στ+1=(1+ϕT​(xτ+1)​Λτ−1​ϕ​(xτ+1))​Σϵ.\\Sigma\_{\\tau+1}=(1+\\phi^{T}(x\_{\\tau+1})\\Lambda^{-1}\_{\\tau}\\phi(x\_{\\tau+1}))\\Sigma\_{\\epsilon}. |  | (8) |

For a derivation of this term, the reader may refer to [minka2000bayesian](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib22)22 2 While the expressions in this work and [minka2000bayesian](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib22) are not obviously equivalent, they can be shown to be the same by applying the Woodbury identity.. Note that several common forms of linear regression, such as ridge regression, are special cases of this general formulation [murphy2012machine](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib23).

Bayesian linear regression has strong connections to kernel methods for GP regression. With the choice of Gaussian priors discussed here, Bayesian linear regression implicitly defines a GP with mean and kernel functions that are functions of the prior on the weights, p⁡(K)p(K), and the basis functions used.

Note that the terms involving the basis functions ϕ\\phi are always weighted inner products of ϕ⁡(x)\\phi(x) and ϕ⁡(x′)\\phi(x^{\\prime}) (with the same weighting, which we will denote Σ\\Sigma). Thus, we define the kernel function k⁡(x,x′)=ϕT​(x)​Σ​ϕ​(x′)k(x,x^{\\prime})=\\phi^{T}(x)\\Sigma\\phi(x^{\\prime}). In contrast to Bayesian linear regression, kernel-based GP regression employs the well-known “kernel trick” to replace occurrences of inner products with the kernel function [rasmussen2004gaussian](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib12). In this way, it can effectively operate in an infinite dimensional feature space such as that defined by the popular squared exponential kernel. This added model capacity comes at the cost of poor computational complexity as the number of samples in the dataset grows.

Figure 1: Overview of the ALPaCA algorithm. Rather than compute the intractable posterior p⁡(yt+1∣xt+1,Dt∗)p(y\_{t+1}\\mid x\_{t+1},D^{\*}\_{t}) directly, we approximate it through a surrogate model for which computing the posterior qξ​(y^t+1∣xt+1,Dt∗)q\_{\\xi}(\\hat{y}\_{t+1}\\mid x\_{t+1},D^{\*}\_{t}) is analytically tractable via Bayesian linear regression. Offline, we optimize the parameters ξ\\xi of this surrogate model on a dataset 𝒟\\mathcal{D} of functions drawn from a prior distribution, aligning the surrogate posterior with the true empirical posterior in the dataset through a meta-learning procedure. Blue nodes are observed offline, green are online inputs, and purple the targets. White nodes are computed, and gray nodes are not observed.

## 4 Approach

In this work, we choose to represent the approximate density q⁡(y∣xt+1,Dt∗)q(y\\mid x\_{t+1},D^{\*}\_{t}) as a neural network model with a Bayesian last layer, which allows employing Bayesian linear regression to compute q⁡(y∣xt+1,Dt∗)q(y\\mid x\_{t+1},D^{\*}\_{t}) analytically. Section [4.1](https://ar5iv.labs.arxiv.org/html/1807.08912#S4.SS1 "4.1 Algorithmic Preliminaries ‣ 4 Approach ‣ Meta-Learning Priors for Efficient Online Bayesian Regression") provides details of this model and other preliminaries. In Section [4.2](https://ar5iv.labs.arxiv.org/html/1807.08912#S4.SS2 "4.2 ALPaCA Overview ‣ 4 Approach ‣ Meta-Learning Priors for Efficient Online Bayesian Regression"), we outline the full ALPaCA algorithm.

### 4.1 Algorithmic Preliminaries

#### Network Model.

Recall y^t\\hat{y}\_{t} denotes the random variable distributed according to q⁡(y∣x,Dτ∗)q(y\\mid x,D^{\*}\_{\\tau}). In this work we will learn a parametric model for y^t\\hat{y}\_{t},

|  |  |  |  |
|---|---|---|---|
|  | y^t=KT​ϕ​(xt,w)+ϵ,\\hat{y}\_{t}=K^{T}\\phi(x\_{t};w)+\\epsilon, |  | (9) |

where ϕ\\phi is a neural network with output dimensionality nϕn\_{\\phi} and weights ww, ϵ∼𝒩⁡(0,Σϵ)\\epsilon\\sim\\mathcal{N}(0,\\Sigma\_{\\epsilon}), and KK is a matrix which may be thought of as the last layer of the neural network. Learning this model based on a dataset for fixed θ\\theta is relatively simple, and is in line with the majority of the literature on neural network regression models. However, we wish to learn a prior distribution over KK such that given data sampled from ([1](https://ar5iv.labs.arxiv.org/html/1807.08912#S2.E1 "In 2 Problem Formulation ‣ Meta-Learning Priors for Efficient Online Bayesian Regression")) with the test parameter θ∗{\\theta^{\*}}, we can generate accurate Gaussian posteriors. Specifically, we wish to learn the parameters K¯0,Λ0\\bar{K}\_{0},\\Lambda\_{0} of a matrix normal prior p⁡(K)=ℳ​𝒩​(K¯0,Λ0−1,Σϵ)p(K)=\\mathcal{MN}(\\bar{K}\_{0},\\Lambda\_{0}^{-1},\\Sigma\_{\\epsilon}), so that we may apply tools from Bayesian linear regression to analytically compute posterior distributions given online data.

While a fully Bayesian approach would also aim to estimate the posterior over the weights ww, we will assume a delta prior over these weights, and thus they are not updated online. However, note that in comparison to previous work that utilizes networks with Bayesian output layers [snoek2015scalable](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib24); [bauer](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib25), we will not train the network and learn the prior over KK in separate phases, and instead they are learned jointly. This is an important distinction: as opposed to simply learning a network to produce good predictions in expectation (with no context data), and adding in a Bayesian last layer afterwards, we aim to learn a network that is composed of basis functions capable of representing the posterior distribution given any context data. Given fixed basis functions, learning the prior terms in a data-driven fashion is known as empirical Bayes [morris1983parametric](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib26). In contrast to the empirical Bayes approach, we also learn the basis functions in addition to the matrix normal prior. Thus, in summary, the parameters of the posterior predicitve model qξq\_{\\xi} we wish to optimize are ξ=(K¯0,Λ0,w)\\xi=(\\bar{K}\_{0},\\Lambda\_{0},w).

#### Optimization Objective.

We will denote the process generating D∗D^{\*}, data xx, and label yy as p⁡(x,y,D∗∣θ∗)p(x,y,D^{\*}\\mid{\\theta^{\*}}), for online test parameters θ∗{\\theta^{\*}}. Note that this may equivalently be written as p⁡(y∣x,θ∗)​p​(x,D∗∣θ∗)p(y\\mid x,{\\theta^{\*}})p(x,D^{\*}\\mid{\\theta^{\*}}), as yy is conditionally independent of D∗D^{\*}, given θ∗{\\theta^{\*}}. Making explicit the dependence of the posterior on the prior terms, the inner expectation of ([3](https://ar5iv.labs.arxiv.org/html/1807.08912#S2.E3 "In 2 Problem Formulation ‣ Meta-Learning Priors for Efficient Online Bayesian Regression")), wherein the expectation over θ∗{\\theta^{\*}} and tt is temporarily ignored, may be written

|  |  |  |
|---|---|---|
|  | 𝔼x,D∗∼p⁡(x,D∗∣θ∗)\[𝔼y∼p⁡(y∣x,θ∗)\[logp(y∣x,D∗)−logqξ(y∣x,D∗)∣x,D∗]].\\displaystyle\\mathbb{E}\_{x,D^{\*}\\sim p(x,D^{\*}\\mid{\\theta^{\*}})}\\left\[\\mathbb{E}\_{y\\sim p(y\\mid x,{\\theta^{\*}})}\[\\log p(y\\mid x,D^{\*})-\\log q\_{\\xi}(y\\mid x,D^{\*})\\mid x,D^{\*}]\\right]. |  |

We can then write the above divergence, dropping the terms that do not affect the optimization over the prior terms, as

|  |  |  |
|---|---|---|
|  | L⁡(K¯0,Λ0,w,θ∗)=−𝔼x,y,D∗∼p⁡(x,y,D∗∣θ∗)​\[log⁡qξ​(y∣x,D∗)].\\displaystyle L(\\bar{K}\_{0},\\Lambda\_{0},w,{\\theta^{\*}})=-\\mathbb{E}\_{x,y,D^{\*}\\sim p(x,y,D^{\*}\\mid{\\theta^{\*}})}\\left\[\\log q\_{\\xi}(y\\mid x,D^{\*})\\right]. |  |

Rewriting this negative log likelihood using ([7](https://ar5iv.labs.arxiv.org/html/1807.08912#S3.E7 "In 3 Bayesian Regression ‣ Meta-Learning Priors for Efficient Online Bayesian Regression")), discarding terms not relevant to the optimization problem over the prior terms, and making the time dependency explicit, we obtain

|  |  |  |  |
|---|---|---|---|
|  | L\\displaystyle L | (K¯0,Λ0,w,θ∗,t)=\\displaystyle(\\bar{K}\_{0},\\Lambda\_{0},w,{\\theta^{\*}},t)= |  |
|  |  | 𝔼xt+1,yt+1,Dt∗∼p⁡(x,y,D∗∣θ∗)\[logdet(Σt+1)+(yt+1−K¯tTϕt+1)TΣt+1−1(yt+1−K¯tTϕt+1)].\\displaystyle\\mathbb{E}\_{x\_{t+1},y\_{t+1},D^{\*}\_{t}\\sim p(x,y,D^{\*}\\mid{\\theta^{\*}})}\\left\[\\log\\det(\\Sigma\_{t+1})+(y\_{t+1}-\\bar{K}\_{t}^{T}\\phi\_{t+1})^{T}\\Sigma\_{t+1}^{-1}(y\_{t+1}-\\bar{K}\_{t}^{T}\\phi\_{t+1})\\right]. |  |

Thus, ([3](https://ar5iv.labs.arxiv.org/html/1807.08912#S2.E3 "In 2 Problem Formulation ‣ Meta-Learning Priors for Efficient Online Bayesian Regression")) may be written as

|  |  |  |  |
|---|---|---|---|
|  | ℓ⁡(K¯0,Λ0,w)=𝔼t∼p⁡(t),θ∗∼p⁡(θ)​\[L⁡(K¯0,Λ0,w,θ∗,t)]\\ell(\\bar{K}\_{0},\\Lambda\_{0},w)=\\mathbb{E}\_{t\\sim p(t),{\\theta^{\*}}\\sim p(\\theta)}\\left\[L(\\bar{K}\_{0},\\Lambda\_{0},w,{\\theta^{\*}},t)\\right] |  | (10) |

where Λt\\Lambda\_{t}, K¯t\\bar{K}\_{t}, and Σt\\Sigma\_{t} are as in ([5](https://ar5iv.labs.arxiv.org/html/1807.08912#S3.E5 "In 3 Bayesian Regression ‣ Meta-Learning Priors for Efficient Online Bayesian Regression")), ([6](https://ar5iv.labs.arxiv.org/html/1807.08912#S3.E6 "In 3 Bayesian Regression ‣ Meta-Learning Priors for Efficient Online Bayesian Regression")), and ([8](https://ar5iv.labs.arxiv.org/html/1807.08912#S3.E8 "In 3 Bayesian Regression ‣ Meta-Learning Priors for Efficient Online Bayesian Regression")). These terms are functions of the prior terms, K¯0,Λ0\\bar{K}\_{0},\\Lambda\_{0}, and ww, as well as Dt∗D^{\*}\_{t}. Here, we write ϕt\\phi\_{t} to denote ϕ⁡(xt)\\phi(x\_{t}). This equivalence between minimizing KL divergence and maximizing log likelihood is well-known [murphy2012machine](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib23), but we emphasize it to improve clarity regarding the expectations and role of the prior terms in the likelihood.

### 4.2 ALPaCA Overview

Algorithm 1 ALPaCA: Offline

1: training data 𝒟\\mathcal{D}, noise variance Σϵ\\Sigma\_{\\epsilon}

2: Randomly initialize K¯0,Λ0,w\\bar{K}\_{0},\\Lambda\_{0},w

3: while not converged do

4:   for j=1j=1 to JJ do

5:    Sample tjt\_{j} from uniform distribution over {1,…,τ}\\{1,\\ldots,\\tau\\}

6:    Sample dataset D(j)D^{(j)} uniformly from 𝒟\\mathcal{D}

7:    Compute K¯tj,Λtj,Σtj+1\\bar{K}\_{t\_{j}},\\Lambda\_{t\_{j}},\\Sigma\_{t\_{j}+1} via ([12](https://ar5iv.labs.arxiv.org/html/1807.08912#S4.E12 "In Offline Training. ‣ 4.2 ALPaCA Overview ‣ 4 Approach ‣ Meta-Learning Priors for Efficient Online Bayesian Regression")), using dataset Dtj+1(j)∖{(xtj+1,ytj+1)}D^{(j)}\_{t\_{j}+1}\\setminus\\{(x\_{t\_{j}+1},y\_{t\_{j}+1})\\}

8:   end for

9:   Update K0,Λ0,wK\_{0},\\Lambda\_{0},w via gradient step on the optimization problem ([12](https://ar5iv.labs.arxiv.org/html/1807.08912#S4.E12 "In Offline Training. ‣ 4.2 ALPaCA Overview ‣ 4 Approach ‣ Meta-Learning Priors for Efficient Online Bayesian Regression"))

10: end while

11: return K¯0,Λ0,w\\bar{K}\_{0},\\Lambda\_{0},w

#### Offline Training.

Given ([10](https://ar5iv.labs.arxiv.org/html/1807.08912#S4.E10 "In Optimization Objective. ‣ 4.1 Algorithmic Preliminaries ‣ 4 Approach ‣ Meta-Learning Priors for Efficient Online Bayesian Regression")), we are now able to write the full optimization problem we wish to solve. We sample JJ datasets DτD\_{\\tau} of τ\\tau samples each uniformly from 𝒟\\mathcal{D}. Let jj denote the index of these datasets. For each of these datasets, we sample a horizon tj∼p⁡(t)t\_{j}\\sim p(t) and clip each dataset to this horizon to obtain {Dtj+1(j)}j=1J\\{D^{(j)}\_{t\_{j}+1}\\}\_{j=1}^{J}. Because 𝒟\\mathcal{D} contains datasets with θ∼p⁡(θ)\\theta\\sim p(\\theta), evaluating the loss on this minibatch approximates the expectations in ([10](https://ar5iv.labs.arxiv.org/html/1807.08912#S4.E10 "In Optimization Objective. ‣ 4.1 Algorithmic Preliminaries ‣ 4 Approach ‣ Meta-Learning Priors for Efficient Online Bayesian Regression")). We use the superscript (j)(j) to denote the terms within each dataset Dtj+1(j)D^{(j)}\_{t\_{j}+1}. Note that

|  |  |  |  |
|---|---|---|---|
|  | logdet(Σtj+1)\\displaystyle\\log\\det(\\Sigma\_{t\_{j}+1}) | =logdet((1+ϕT(xtj+1)Λtj−1ϕ(xtj+1))Σϵ)\\displaystyle=\\log\\det((1+\\phi^{T}(x\_{t\_{j}+1})\\Lambda^{-1}\_{t\_{j}}\\phi(x\_{{t\_{j}+1}}))\\Sigma\_{\\epsilon}) |  |
|  |  | =nylog(1+ϕT(xtj+1)Λtj−1ϕ(xtj+1))+logdetΣϵ.\\displaystyle={n\_{y}}\\log(1+\\phi^{T}(x\_{{t\_{j}+1}})\\Lambda^{-1}\_{t\_{j}}\\phi(x\_{{t\_{j}+1}}))+\\log\\det\\Sigma\_{\\epsilon}. |  |

Since it does not depend on the optimization variables, logdetΣϵ\\log\\det\\Sigma\_{\\epsilon} may be ignored and we may write the Monte Carlo estimate of ([10](https://ar5iv.labs.arxiv.org/html/1807.08912#S4.E10 "In Optimization Objective. ‣ 4.1 Algorithmic Preliminaries ‣ 4 Approach ‣ Meta-Learning Priors for Efficient Online Bayesian Regression")) as

|  |  |  |  |  |
|---|---|---|---|---|
|  | ℓ^​(K¯0,Λ0,w)≔1J​∑j=1J(CLOSE\\displaystyle\\hat{\\ell}(\\bar{K}\_{0},\\Lambda\_{0},w)\\coloneqq\\frac{1}{J}\\sum\_{j=1}^{J}\\biggl( | ny​log⁡(1+ϕtj+1(j),T​Λtj(j),−1​ϕtj+1(j))\\displaystyle{\\displaystyle n\_{y}}\\log\\left(1+\\phi^{(j),T}\_{t\_{j}+1}\\Lambda^{(j),-1}\_{t\_{j}}\\phi^{(j)}\_{t\_{j}+1}\\right) |  | (11) |
|  |  | OPEN+(ytj+1(j)−K¯tj(j),T​ϕtj+1(j))T​Σtj+1(j),−1​(ytj+1(j)−K¯tj(j),T​ϕtj+1(j))).\\displaystyle+\\left(y^{(j)}\_{t\_{j}+1}-\\bar{K}^{(j),T}\_{{t\_{j}}}\\phi^{(j)}\_{{t\_{j}}+1}\\right)^{T}\\Sigma\_{t\_{j}+1}^{(j),-1}\\left(y^{(j)}\_{t\_{j}+1}-\\bar{K}\_{{t\_{j}}}^{(j),T}\\phi^{(j)}\_{{t\_{j}}+1}\\right)\\biggr). |  |  |

Thus, our problem formulation is

|  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|
|  | \\displaystyle | minK¯0,Λ0,w\\displaystyle\\underset{\\bar{K}\_{0},\\Lambda\_{0},w}{\\min} |  | ℓ^​(K¯0,Λ0,w)\\displaystyle\\hat{\\ell}(\\bar{K}\_{0},\\Lambda\_{0},w) |  | (12) |
|  |  | s.t.\\displaystyle\\text{\\,\\,\\, s.t.} |  | Σ(j)tj+1=(1+ϕT(x(j)tj+1)Λ(j),−1tjϕ(x(j)tj+1))Σϵ,j=1,…,J\\displaystyle\\Sigma^{(j)}\_{{t\_{j}}+1}=(1+\\phi^{T}(x^{(j)}\_{{t\_{j}}+1})\\Lambda^{(j),-1}\_{t\_{j}}\\phi(x^{(j)}\_{{t\_{j}}+1}))\\Sigma\_{\\epsilon},\\,\\,j=1,\\ldots,J |  |  |
|  |  | Λ(j)tj=Φ(j),TtjΦ(j)tj+Λ0,j=1,…,J\\displaystyle\\Lambda^{(j)}\_{t\_{j}}=\\Phi^{(j),T}\_{t\_{j}}\\Phi^{(j)}\_{t\_{j}}+\\Lambda\_{0},\\,\\,j=1,\\ldots,J |  |  |  |  |
|  |  | K¯(j)tj=(Λ(j)tj)−1(Φtj(j),TYtj(j)+Λ0K¯0),j=1,…,J\\displaystyle\\bar{K}^{(j)}\_{t\_{j}}=(\\Lambda^{(j)}\_{t\_{j}})^{-1}(\\Phi\_{t\_{j}}^{(j),T}Y\_{t\_{j}}^{(j)}+\\Lambda\_{0}\\bar{K}\_{0}),\\,\\,j=1,\\ldots,J |  |  |  |  |
|  |  | Λ0⪰0\\displaystyle\\Lambda\_{0}\\succeq 0 |  |  |  |  |

where Φtj(j)\\Phi^{(j)}\_{t\_{j}} and Ytj(j)Y^{(j)}\_{t\_{j}} contain data from dataset jj, from timesteps 1,…,tj1,\\ldots,t\_{j}. The full algorithm is detailed in Algorithm [1](https://ar5iv.labs.arxiv.org/html/1807.08912#alg1 "Algorithm 1 ‣ 4.2 ALPaCA Overview ‣ 4 Approach ‣ Meta-Learning Priors for Efficient Online Bayesian Regression"). The semidefinite constraint on Λ0\\Lambda\_{0} is easily enforced by optimizing instead over L0L\_{0}, the Cholesky decomposition (Λ0=L0​L0T\\Lambda\_{0}=L\_{0}L\_{0}^{T}). By using this Cholesky decomposition and substituting in the equality constraints, ([12](https://ar5iv.labs.arxiv.org/html/1807.08912#S4.E12 "In Offline Training. ‣ 4.2 ALPaCA Overview ‣ 4 Approach ‣ Meta-Learning Priors for Efficient Online Bayesian Regression")) is made unconstrained, and may be solved by stochastic gradient descent. Note that in practice, rather than evaluating the loss only on (xtj+1,ytj+1)(x\_{t\_{j}+1},y\_{t\_{j}+1}), we minimize the average loss evaluated on {(xt,yt)}t=tj+1τ\\{(x\_{t},y\_{t})\\}\_{t=t\_{j}+1}^{\\tau}.

#### Online Phase.

Algorithm 2 ALPaCA: Online

1: prior terms Λ0,K¯0,w\\Lambda\_{0},\\bar{K}\_{0},w, noise variance Σϵ\\Sigma\_{\\epsilon}, data {(xt,yt)}t=1τ\\{(x\_{t},y\_{t})\\}\_{t=1}^{\\tau}

2: Q0←Λ0​K¯0Q\_{0}\\leftarrow\\Lambda\_{0}\\bar{K}\_{0}

3: Compute Λ0−1\\Lambda\_{0}^{-1}

4: for t=1,…,τt=1,\\ldots,\\tau do

5:   Λt−1←Λt−1−1−11+ϕtT​Λt−1−1​ϕt​(Λt−1−1​ϕt)​(Λt−1−1​ϕt)T\\Lambda\_{t}^{-1}\\leftarrow\\Lambda\_{t-1}^{-1}-\\frac{1}{1+\\phi\_{t}^{T}\\Lambda\_{t-1}^{-1}\\phi\_{t}}(\\Lambda\_{t-1}^{-1}\\phi\_{t})(\\Lambda\_{t-1}^{-1}\\phi\_{t})^{T}

6:   Qt←ϕt​ytT+Qt−1Q\_{t}\\leftarrow\\phi\_{t}y\_{t}^{T}+Q\_{t-1}

7:   K¯t←Λt−1​Qt\\bar{K}\_{t}\\leftarrow\\Lambda\_{t}^{-1}Q\_{t}

8:   y¯t+1←KtT​ϕt+1\\bar{y}\_{t+1}\\leftarrow K\_{t}^{T}\\phi\_{t+1}

9:   Σt+1←(1+ϕt+1T​Λt−1​ϕt+1)​Σϵ\\Sigma\_{t+1}\\leftarrow(1+\\phi^{T}\_{t+1}\\Lambda^{-1}\_{t}\\phi\_{t+1})\\Sigma\_{\\epsilon}

10: end for

11: return y¯t+1,Σt+1\\bar{y}\_{t+1},\\Sigma\_{t+1}

After learning the prior parameters (K¯0,Λ0)(\\bar{K}\_{0},\\Lambda\_{0}) and the weights for the basis functions ww, ALPaCA computes the online posterior by performing Bayesian linear regression using the data observed online D∗D^{\*}. Since the number of basis functions used may be large for complex functions, we introduce a recursive Bayesian update that improves computational efficiency. This result is well known in linear system identification and online regression [ljung1983theory](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib27). Let Q0≔Λ0​K¯0Q\_{0}\\coloneqq\\Lambda\_{0}\\bar{K}\_{0}. Then, the online updates may be written as

|  |  |  |  |  |
|---|---|---|---|---|
|  | Λt−1\\displaystyle\\Lambda\_{t}^{-1} | =Λt−1−1−11+ϕtT​Λt−1−1​ϕt​(Λt−1−1​ϕt)​(Λt−1−1​ϕt)T\\displaystyle=\\Lambda^{-1}\_{t-1}-\\frac{1}{1+\\phi\_{t}^{T}\\Lambda\_{t-1}^{-1}\\phi\_{t}}(\\Lambda\_{t-1}^{-1}\\phi\_{t})(\\Lambda\_{t-1}^{-1}\\phi\_{t})^{T} |  | (13) |
|  | Qt\\displaystyle Q\_{t} | =ϕt​ytT+Qt−1\\displaystyle=\\phi\_{t}y\_{t}^{T}+Q\_{t-1} |  | (14) |

where ϕt=ϕ⁡(xt)\\phi\_{t}=\\phi(x\_{t}). The first update comes from the Woodbury identity [ljung1983theory](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib27). Importantly, this recursive update rule decreases the complexity from the original formulation’s 𝒪⁡(nϕ3)\\mathcal{O}(n\_{\\phi}^{3}) (resulting from inverting Λn\\Lambda\_{n} directly) to 𝒪⁡(nϕ2)\\mathcal{O}(n\_{\\phi}^{2}). The full recursive online algorithm is presented in Algorithm [2](https://ar5iv.labs.arxiv.org/html/1807.08912#alg2 "Algorithm 2 ‣ Online Phase. ‣ 4.2 ALPaCA Overview ‣ 4 Approach ‣ Meta-Learning Priors for Efficient Online Bayesian Regression").

Note that ϕ\\phi is not updated online, in contrast to some other approaches to meta-learning such as MAML [finn2017model](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib21). Moving from gradient updates to least squares yields a simpler algorithm online, which is useful for debugging or verifying autonomous systems. Because ALPaCA performs GP regression in the weight space as opposed to computing the kernel functions, it achieves linear complexity in the amount of data. In particular, for nn context data points and mm test data points, ALPaCA has complexity 𝒪⁡(n+m)\\mathcal{O}(n+m). One interpretation of ALPaCA is as an approximation method for kernel-based GP regression, similar to randomized kernels [rahimi2008random](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib20). However, instead of randomizing basis functions to approximate well-known kernels, we compute basis functions to maximize predictive power with respect to the prior implicit in the training data.

## 5 Related Work

Thus far we have introduced ALPaCA in the context of GP regression. In this section, we discuss connections to other related works in meta-learning and sample efficient regression.

#### Meta-Learning.

In this work, we present a Bayesian perspective on meta-learning. Recently, model-agnostic meta-learning (MAML) presented a simple approach to meta-learning with impressive performance. MAML was originally introduced from the perspective of learning an initial set of neural network weights that could efficiently reach the desired weights for a task [finn2017model](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib21), given a handful of gradient steps on samples from that task. Subsequent work then recast this as a form of hierarchical Bayes [grant2018recasting](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib28). Specifically, indexing tasks with jj, the marginal likelihood of the data XX is written

|  |  |  |
|---|---|---|
|  | p⁡(X∣w)=∏j(∫p⁡(xj1,…,xjτ∣wj′)​p​(wj′∣w)​d​wj′),p(X\\mid w)=\\prod\_{j}\\left(\\int p(x\_{j\_{1}},\\ldots,x\_{j\_{\\tau}}\\mid w\_{j}^{\\prime})p(w\_{j}^{\\prime}\\mid w)dw\_{j}^{\\prime}\\right), |  |

where xjtx\_{j\_{t}} denote data points from within task jj, wj′w\_{j}^{\\prime} denotes the post meta-update weights for task and loss function jj, and ww denote the initial weights. They note that MAML estimates the likelihood of the data drawn from task jj with a point estimate of wj′w\_{j}^{\\prime}, computed via gradient descent. In the case of linear regression, Santos [santos1996equivalence](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib29) showed that early stopping of gradient descent to compute wj′w\_{j}^{\\prime} results in a Maximum a Posteriori (MAP) estimate of wj′w\_{j}^{\\prime}, regularized by ‖wj′−w‖Q2\\|w\_{j}^{\\prime}-w\\|^{2}\_{Q}, where QQ depends on the gradient step size and the number of iterations. In contrast to these formulations, we assume a delta prior on the network weights, and so the weights are not updated online. Additionally, our formulation of a Bayesian last layer allows the posterior (given the context data) to be computed directly. Because we perform this analytical Bayesian update, we do not suffer the effects of the regularizing term, and instead may exactly compute the Gaussian approximation of the posterior. Generally, in the case of linear filtering, recursive least squares approaches have been shown to outperform gradient-based approaches [cioffi1984fast](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib30). These results from the linear case may explain in part why ALPaCA makes more efficient use of the online data provided, and can perform faster updates.

The interpretation of MAML as hierarchical Bayes allowed the authors to incorporate Laplace approximation into the weight update [grant2018recasting](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib28), and the authors refer to the resulting algorithm as LLAMA. This approximation results in the point estimate of the updated weights being replaced with a second-order approximation of the posterior. Thus, sampling based methods can be used to approximate the posterior distribution of the data. More recently, PLATIPUS [finn2018probabilistic](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib31) and Bayesian MAML [kim2018bayesian](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib32) have used variational approaches to move from point estimates to Bayesian posteriors. We wish to emphasize that these approaches are capable of capturing a larger family of distributions (such as multimodal distributions), while we achieve improved sample efficiency and better efficiency in terms of computation. We believe these trade-offs are justified and necessary in many robotics applications.

#### Efficient Stochastic Process Regression.

In this work we have considered meta-learning for GP regression. A relaxation of this problem formulation to estimation of general stochastic processes was recently presented in [np2018](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib19). This approach is based on amortized variational inference (as in PLATIPUS [finn2018probabilistic](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib31)). In particular, it is a form of conditional variational autoencoder (CVAE) [kingma2013auto](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib33); [yan2016attribute2image](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib34), where the conditioning variable is a function of the context data. As with other autoencoder-based approaches, this method requires sampling-based estimates of the moments of the distribution. In contrast, in ALPaCA, mean and variance may be computed analytically. Moreover, [np2018](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib19) was only tested on problems with two dimensional output spaces, and it is unclear how well it performs for larger problems, whereas ALPaCA performs well even for high dimensional systems (such as the 12D hopper system). Within the context of meta-learning for GPs, [yu2005learning](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib35) took a similar approach to the one presented here. However, while they are learning hyperparameter priors based on a distribution over tasks, they do not jointly learn a set of basis functions capable of accurately representing posterior distributions.

#### Bayesian Output Layer Models.

The approach of affixing a Bayesian last layer to a standard neural network has seen application across machine learning. In [snoek2015scalable](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib24), the authors use this network structure with an arbitrary prior for Bayesian optimization, and [azizzadenesheli2018efficient](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib36) use it to improve exploration efficiency in reinforcement learning. In the context of meta-learning, [bauer](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib25) use a similar approach to the one presented here. However, they first train their network without considering training it with respect to meta-updated last layers and losses, as we have here. They then perform meta-learning by updating the last layer using approximate methods. In contrast to this approach, we learn network weights such that they form useful basis functions for estimating the posterior, for any amount of data accrued online. Our work also shares common features with recent work on meta-learning with closed form solvers [bertinetto2018meta](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib37), in which the authors do meta-learning with a ridge regression last layer. Generally speaking, these approaches all make arbitrary assumptions regarding the output layer prior, whereas we map our prior information (available via the training data) to a prior over weights.

## 6 Experiments

Figure 2: Comparison of ALPaCA (left) compared to ALPaCA without meta-training (center) and Gaussian process regression (right) on a simple sinusoid problem from [finn2017model](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib21). The plots on each row contain 0, 1, 2, 3, and 5 context samples respectively. Confidence intervals are 95%.

Figure 3: Comparison of ALPaCA (left) compared to ALPaCA without meta-training (center) and Gaussian process regression (right) on a function exhibiting discrete switching between y=−1y=-1 and 11 for x∈\[−2.5,2.5]x\\in\[-2.5,2.5]. Three switching points are randomly sampled. The plots on each row contain 0, 5, 10, 15, and 25 context samples.

In this section, we investigate the performance of ALPaCA on a variety of regression problems, ranging from simple toy problems which help in building intuition for the algorithm, to challenging prediction and dynamics modeling problems. We compare ALPaCA against several other algorithms that are capable of rapid online adaptation. First, as an ablation experiment, we compare to ALPaCA without the meta-training. This is to say, the network is trained to predict the prior for varying datasets, but is not explicitly conditioned on random selections of data. This is roughly equivalent to standard methods for training neural networks with Bayesian last layers [snoek2015scalable](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib24); [azizzadenesheli2018efficient](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib36). In these works, a network is typically trained to minimize e.g. MSE loss over all of the data, and then the prior over the last layer is chosen arbitrarily. For the dynamical systems that we investigate, we also compare to the predictive performance of ALPaCA with no online updating. This network is just trained in expectation over the sampled values of the model parameters θ\\theta. For our toy examples, we compare to GP regression. We chose a zero-mean GP with a squared-exponential kernel, which is often the default choice in GP modeling [snoek2012practical](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib11). The three above approaches are compared in terms of negative log likelihood to compare the accuracy of their posterior predictions. In our experiments we use tanh nonlinearities, due to favorable properties in terms of variance smoothness and the behavior of the variance far from the observed data. This is discussed in more detail in the appendix.

In addition to these comparisons, we also compare to MAML [finn2017model](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib21), which has rapidly become one of the most common approaches for meta-learning. Indeed, MAML was recently used for online system identification [clavera2018learning](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib38) on relatively complex dynamical systems. Because MAML generates point predictions rather than full posteriors, we compare in terms of MSE loss (for ALPaCA, we compute MSE loss on the mean of the posterior). We were not able to compare to PLATIPUS [finn2018probabilistic](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib31) or Neural Processes [np2018](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib19) because to our knowledge, at the time of writing, the code is not publicly available. Throughout our experiments, plotted confidence intervals are 95%.

### 6.1 Toy Experiments

We first investigate two simple regression examples that allow both quantitative and qualitative evaluation of performance. First, we investigate a regression problem over a family of sinusoids, where the amplitude and phase were sampled uniformly from \[0.1,5.0]\[0.1,5.0] and \[0,π]\[0,\\pi] respectively. This sinusoid problem was used as a simple test problem in [finn2017model](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib21). We also evaluate performance on a new, simple benchmark that consists of a family of two-valued functions that begin at y=−1y=-1 for x&lt;2.5x&lt;2.5, end at y=1y=1 for x&gt;2.5x&gt;2.5, and switch three times between these values for x∈\[−2.5,2.5]x\\in\[-2.5,2.5]. These switching points are sampled uniformly from this range. This function was designed to be challenging relative to its apparent simplicity for our approach. Because we are performing basis function regression, the discrete switches between −1-1 and 11 which can occur anywhere within the range specified above, are hard to capture. This is also challenging for GP approaches, as the structure of the function is hard to capture via a single global smoothness parameter in the kernel. For example, in [SvenssonSchon2017](https://ar5iv.labs.arxiv.org/html/1807.08912#bib.bib39), the authors learn separate GPs for each piecewise continuous segment, as opposed to loosening their smoothness priors.

The outputs of ALPaCA, ALPaCA without meta-training, and Gaussian process regression (GPR) on these two problems are plotted in figures [2](https://ar5iv.labs.arxiv.org/html/1807.08912#S6.F2 "Figure 2 ‣ 6 Experiments ‣ Meta-Learning Priors for Efficient Online Bayesian Regression") and [3](https://ar5iv.labs.arxiv.org/html/1807.08912#S6.F3 "Figure 3 ‣ 6 Experiments ‣ Meta-Learning Priors for Efficient Online Bayesian Regression"). First, note that ALPaCA is capable of making predictions that incorporate structural knowledge of the problem from a single datapoint, compared to the local predictions that are common for GPs with the SE kernel. Secondly, on the sinusoid problem, the first sample allows ALPaCA to correctly reduce uncertainty at points at a distance of a scalar multiple of a half wavelength from the sampled point. Moreover, one sample is enough to have a good estimate of the sinusoid nearly everywhere, and within five samples the estimated variance has dropped to nearly zero. This reduction in variance is observed for ALPaCA without meta-training as well, but its predictions are incorrect, highlighting the utility of our meta-learning approach. Further experiments with varying noise covariance ar

...(content truncated)
