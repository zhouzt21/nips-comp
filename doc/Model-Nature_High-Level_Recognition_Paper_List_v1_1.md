# Model-Nature High-Level Recognition — Paper List v1

> **状态**：Framework v0 配套的 Model-Nature 认识积累库；本文件不是新版 Framework，不修改 `Research Down–Up Loop(1).md`。  
> **时间**：2026-10-16  
> **本轮目标**：在既有 MN-P01–MN-P44 的基础上，继续寻找跨 model-family 的 high-level recognition，尤其加深三个关键节点：
>
> 1. **MN-09 Architecture / Inductive Bias**
> 2. **MN-03 Training Dynamics**
> 3. **MN-07 Representation Geometry**
>
> 同时继续补充其与 State/Memory、Knowledge、Circuits、Context Access、ICL、World/Latent Semantics、Expressivity、Data、Scale、Objective 等节点之间的认识依赖。
>
> **证据约定**
>
> - **Strong**：论文直接给出理论、跨架构实验或明确机制证据，可支持该认识作为研究先验。
> - **Medium**：有较强实证/理论支持，但依赖特定 model family、task 或分析设定。
> - **Open**：值得作为 high-level hypothesis，但现有证据不足以把它当成普遍规律。
> - 本文件中的“在图中的作用”是对论文贡献的 **model-nature abstraction**，不是论文作者原话；应在后续精读时回到原文核验。

---

# 0. 当前 Model-Nature Recognition Graph v0.1

当前不再把 16 个节点理解成平行目录，而暂时视为不同层级的认识节点：

```text
                    SUBSTRATE / INTERFACE
              ┌──────────────┴──────────────┐
              │                             │
      Tokenization / Interface      Architecture / Inductive Bias
              │                       │       │        │
              │                       │       │        └────► Expressivity / Limits
              │                       │       │
              │                       │       └────► Representation Geometry
              │                       │                    │
              │                       │                    ├────► State / Memory
              │                       │                    ├────► Parametric Knowledge
              │                       │                    ├────► Circuits / Composition
              │                       │                    └────► World / Latent Semantics
              │                       │
              │                       └────► Conditional Computation / Routing
              │
              ▼
       Representation Possibility


 DATA + SCALE + OBJECTIVE
          │       │       │
          └───────┼───────┘
                  ▼
          Training Dynamics
                  │
          ┌───────┴────────┐
          ▼                ▼
 Representation       Runtime Computation
   Geometry           / Adaptation
          │                │
          ├──────► Circuits
          │                │
          ├──────► Memory   ├────► ICL
          │                │
          ├──────► Knowledge
          │                └────► Test-Time Compute
          │
          └────► World / Latent Semantics
                           │
                           ▼
                       Capability
```

## 当前最重要的结构判断

### Hub A — Architecture / Inductive Bias

它首先决定：

> **什么样的函数、关系、局部性、对称性、状态更新或信息访问方式，在这个 substrate 中是“自然的”。**

因此 Architecture 同时连接：

\[
\text{Architecture}
\rightarrow
\text{Expressivity}
\]

\[
\text{Architecture}
\rightarrow
\text{Representation Geometry}
\]

\[
\text{Architecture}
\rightarrow
\text{State / Access / Computation}
\]

而不是只对应“模型结构”。

---

### Hub B — Training Dynamics

它回答：

> **既定的 architecture + data + objective，训练过程究竟会把模型推向什么内部组织？**

因此：

\[
\text{Data + Objective + Scale}
\rightarrow
\text{Training Dynamics}
\rightarrow
\text{Representation / Circuit / Knowledge}
\]

尤其重要的是区分：

\[
\text{optimization dynamics}
\neq
\text{representation formation}
\]

但两者又密切耦合。

---

### Hub C — Representation Geometry

它是目前最值得长期积累的中介认识之一。

它不只是：

> “hidden state 在空间里长什么样？”

而更接近：

> **模型把 feature、concept、state、memory、function、knowledge 放进怎样的几何组织中，以及这些几何结构是否能被计算过程实际读取、组合和改变。**

因此它同时连接：

\[
\text{Architecture}
\rightarrow
\text{Geometry}
\]

\[
\text{Training Dynamics}
\rightarrow
\text{Geometry}
\]

以及：

\[
\text{Geometry}
\leftrightarrow
\text{Memory}
\]

\[
\text{Geometry}
\leftrightarrow
\text{Circuits}
\]

\[
\text{Geometry}
\rightarrow
\text{Concept / Knowledge / Function}
\]

\[
\text{Geometry}
\rightarrow
\text{World / Latent Semantics}
\]

但必须保留：

\[
\boxed{\text{Decodable Geometry} \neq \text{Used Geometry}}
\]

---

# 1. Existing Paper List — MN-P01–MN-P44

以下条目承接上一版 paper list，**不改变原有 ID 与节点映射**。

| ID | Priority | Paper | Node | Model family | 在图中的作用 |
|---|---|---|---|---|---|
| MN-P01 | **P0** | *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges* (2021) | MN-09 Architecture / Inductive Bias | **CNN / RNN / GNN / Transformer** | Architecture → geometric inductive bias 的总入口 |
| MN-P02 | P1 | *Deep Sets* (2017) | MN-09, MN-16 | **Set architecture** | 结构约束如何直接写进 architecture |
| MN-P03 | **P0** | *How Powerful are Graph Neural Networks?* (2019) | MN-16 Expressivity | **GNN** | architecture → representational equivalence / hard limit |
| MN-P04 | P1 | *Are Transformers Universal Approximators of Sequence-to-Sequence Functions?* (2020) | MN-16 | **Transformer** | Transformer expressivity 上界/能力 |
| MN-P05 | **P0** | *ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases* (2021) | MN-09, MN-07 | **CNN / ViT** | inductive bias → learned representation / sample efficiency |
| MN-P06 | P1 | *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023) | MN-09, MN-05 | **SSM** | architecture 如何定义 state / propagation / forgetting |
| MN-P07 | **P0** | *Understanding the Differences in Foundation Models: Attention, State Space Models, and Recurrent Neural Networks* (2024) | MN-09, MN-05, MN-16 | **Attention / SSM / RNN** | 跨 architecture 的统一比较框架 |
| MN-P08 | **P0** | *A Comparative Analysis of Attention versus State-Space Models for In-Context Learning* (2026) | MN-05, MN-07, MN-09, MN-04 | **Transformer / Mamba-2** | belief geometry → maintenance / addressing；关键最新桥梁 |
| MN-P09 | P1 | *Tokenization Matters! Degrading Large Language Models through Challenging Their Tokenization* (2024) | MN-13 | **GPT-4o / Llama-3 / Qwen2.5-max 等 LLM** | Interface → capability failure |
| MN-P10 | **P0** | *How Tokenization Limits Phonological Knowledge Representation in Language Models and How to Improve Them* (2026) | MN-13, MN-07 | **Text LMs** | Interface → representation quality |
| MN-P11 | **P0** | *Scaling Laws for Neural Language Models* (2020) | MN-01 Scaling | **Autoregressive LMs** | scale/data/compute → capability / training |
| MN-P12 | P1 | *Training Compute-Optimal Large Language Models* (2022) | MN-01, MN-02 | **Transformer LMs** | scale ↔ data allocation |
| MN-P13 | P1 | *Scaling Laws for Transfer* (2021) | MN-01, MN-02 | **Transformer LMs** | pretraining scale → transferable information |
| MN-P14 | **P0** | *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets* (2022) | MN-03 Training Dynamics | **Small Transformers** | capability emergence ≠ monotonic loss reduction |
| MN-P15 | **P0** | *Progress Measures for Grokking via Mechanistic Interpretability* (2023) | MN-03, MN-08 | **Small Transformers** | training dynamics → circuit formation |
| MN-P16 | **P0** | *On the Origins of Linear Representations in Large Language Models* (2024) | MN-07, MN-14 | **Latent-variable theory + LLaMA-2** | objective + data structure → representation geometry |
| MN-P17 | P1 | *Visualizing and Measuring the Geometry of BERT* (2019) | MN-07 | **BERT** | 直接研究 representation geometry |
| MN-P18 | P1 | *How Contextual are Contextualized Word Representations?* (2019) | MN-07 | **ELMo / BERT / GPT-2** | 跨 architecture 比较 representation geometry |
| MN-P19 | **P0** | *The Platonic Representation Hypothesis* (2024) | MN-07 | **Vision / Language models** | 跨 modality / model 的 representation convergence |
| MN-P20 | **P0** | *A Mathematical Framework for Transformer Circuits* (2021) | MN-07, MN-08 | **Transformer** | representation subspace ↔ computational circuit |
| MN-P21 | **P0** | *In-context Learning and Induction Heads* (2022) | MN-03, MN-08, MN-04 | **Transformer** | training dynamics → circuit → ICL |
| MN-P22 | P1 | *Eliciting Latent Predictions from Transformers with the Tuned Lens* (2023) | MN-07, MN-08 | **Autoregressive Transformer LMs** | representation trajectory → iterative computation |
| MN-P23 | **P0** | *Locating and Editing Factual Associations in GPT* (2022) | MN-06, MN-08 | **GPT / autoregressive Transformer** | parametric knowledge → localized computation |
| MN-P24 | **P0** | *Dissecting Recall of Factual Associations in Auto-Regressive Language Models* (2023) | MN-06, MN-08, MN-12 | **Transformer LM** | knowledge storage → retrieval computation |
| MN-P25 | **P0** | *Implicit Representations of Meaning in Neural Language Models* (2021) | MN-05, MN-07, MN-15 | **BART / T5** | dynamic semantic state |
| MN-P26 | **P0** | *Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task* (2022) | MN-07, MN-05, MN-15 | **Othello-GPT / Transformer** | state representation + causal intervention |
| MN-P27 | **P0** | *Language Models Represent Space and Time* (2024) | MN-07, MN-15 | **Llama-2 / Pythia** | representation geometry → world structure |
| MN-P28 | P1 | *World Models* (2018) | MN-15 | **VAE + MDN-RNN / RL** | 非语言模型的 world-model substrate 对照组 |
| MN-P29 | P1 | *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context* (2019) | MN-05 | **Transformer** | recurrent memory / long-context state |
| MN-P30 | P1 | *Compressive Transformers for Long-Range Sequence Modelling* (2019) | MN-05 | **Transformer** | memory compression |
| MN-P31 | **P0** | *Lost in the Middle: How Language Models Use Long Contexts* (2024) | MN-12 | **GPT-3.5 / Claude / MPT / LongChat / encoder-decoder** | context capacity ≠ context access |
| MN-P32 | **P0** | *Transformers Generalize Differently from Information Stored in Context vs in Weights* (2022) | MN-06, MN-12, MN-04 | **Transformer** | parametric vs contextual information |
| MN-P33 | P1 | *What Learning Algorithm Is In-Context Learning?* (2023) | MN-04, MN-07, MN-08 | **Transformer** | context → implicit learner |
| MN-P34 | P1 | *Transformers Learn In-Context by Gradient Descent* (2023) | MN-04, MN-08 | **Transformer** | runtime learning as forward-pass computation |
| MN-P35 | P1 | *When Context Leads but Parametric Memory Follows in Large Language Models* (2024) | MN-06, MN-12 | **9 LLMs** | context / parametric memory allocation |
| MN-P36 | P1 | *Deciphering the Interplay of Parametric and Non-parametric Memory in Retrieval-augmented Language Models* (2024) | MN-06, MN-08, MN-12 | **Atlas / RAG** | memory-source selection + internal mediation |
| MN-P37 | P1 | *Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters* (2024) | MN-11 | **LLMs** | runtime compute → capability |
| MN-P38 | P1 | *Training Language Models to Follow Instructions with Human Feedback* (2022) | MN-14 | **GPT-3 / InstructGPT** | objective/post-training → behavior |
| MN-P39 | P1 | *Let's Verify Step by Step* (2023) | MN-14, MN-11 | **LLMs** | supervisory granularity → reasoning behavior |
| MN-P40 | P1 | *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* (2021) | MN-10, MN-09 | **Sparse MoE Transformer** | architecture → conditional computation / routing |
| MN-P41 | **P0 / recent** | *Attractor Geometry of Transformer Memory: From Conflict Arbitration to Confident Hallucination* (2026) | MN-07, MN-05, MN-06, MN-12 | **Autoregressive Transformer** | **Geometry ↔ memory ↔ knowledge conflict** |
| MN-P42 | **P0 / recent** | *Understanding and Controlling the Geometry of Memory Organization in RNNs* (2025) | MN-07, MN-05 | **RNN / GRU / LSTM-style memory** | **Geometry → temporal memory organization** |
| MN-P43 | P1 / recent | *A Phenomenology of Neural Geometry in Vision-Language-Action Models* (2026) | MN-07 | **VLA** | geometry 不只是可读性，而有可干预的形状 |
| MN-P44 | P1 | *Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation* (2025) | MN-15 | **VLM** | world-model hypothesis 的系统化 evaluation |

---

# 2. New Additions — MN-P45–MN-P65

## A. Architecture / Inductive Bias — 深化 MN-09

| ID | Priority | Paper | Node | Model family | 在图中的作用 | Evidence |
|---|---|---|---|---|---|---|
| MN-P45 | **P0** | *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention* (2020) | MN-09, MN-05, MN-12 | **Transformer / RNN-like linear attention** | 说明“attention architecture”和“recurrent state update”并非完全不同的计算范式；architecture 可以改变 state 的显式组织方式 | **Strong** |
| MN-P46 | **P0** | *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (2024) | MN-09, MN-05, MN-08 | **Transformer / SSM** | 进一步把 Transformer 与 SSM 放进共同的 structured state-space / matrix decomposition 视角；architecture family 的边界本身可能不是最基本的解释单位 | **Strong** |
| MN-P47 | **P0** | *Does Syntax Need to Grow on Trees? Sources of Hierarchical Inductive Bias in Sequence-to-Sequence Networks* (2020) | MN-09, MN-16 | **Tree model / RNN / GRU / LSTM** | 同一数据、同一任务下，不同 architecture 会偏向不同结构性 generalization；architecture 是 hypothesis space 的偏置来源，而非单纯容量 | **Strong** |
| MN-P48 | **P0 / recent** | *The Inductive Bias of Convolutional Neural Networks: Locality and Weight Sharing Reshape Implicit Regularization* (2026) | MN-09, MN-03, MN-07 | **CNN / fully-connected** | 把 locality + weight sharing 与 implicit regularization、patch geometry、generalization 直接连起来；architecture 不只是“表达什么”，还改变训练时偏向什么 | **Strong** |
| MN-P49 | P1 | *On the Inductive Bias of Neural Tangent Kernels* (2019) | MN-09, MN-03, MN-16 | **CNN / NTK regime** | 将 architecture 的 inductive bias 放到 induced function space / RKHS 中理解；架构偏置可以表现为“更偏好哪类函数” | **Strong** |
| MN-P50 | P1 | *On the Inductive Bias of Neural Networks for Learning Read-Once DNFs* (2022) | MN-09, MN-03 | **Convex neural network** | 即使存在大量 zero-error solutions，gradient methods 仍偏向与目标逻辑结构对齐的解；architecture + optimization 共同形成 solution bias | **Medium–Strong** |
| MN-P51 | P1 / recent | *Learning High-Dimensional Parity Functions with Product Networks using Gradient Descent* (2026) | MN-09, MN-16, MN-03 | **Product networks / MLP baselines** | architecture 与 data sparsity 联合决定某类结构化任务是否可高效学习；提示“任务难度”必须相对于 substrate 来定义 | **Medium–Strong** |

### 对 MN-09 的新认识

从这些工作开始，`Architecture / Inductive Bias` 最好不再理解成：

> “CNN 有 locality，Transformer 有 global attention，SSM 有 recurrence。”

更高层的认识是：

\[
\boxed{
Architecture
\rightarrow
\begin{cases}
\text{expressible functions}\\
\text{available computational primitives}\\
\text{information-access pattern}\\
\text{state-update mechanism}\\
\text{implicit regularization}\\
\text{representation geometry}
\end{cases}
}
\]

因此，一个实验观察到“模型不会 X”时，不能只问：

> 参数够不够？

还应该问：

> **这个 substrate 是否天然提供了表达、存储、访问、更新 X 所需要的 primitive？**

---

# 3. Training Dynamics — 深化 MN-03

| ID | Priority | Paper | Node | Model family | 在图中的作用 | Evidence |
|---|---|---|---|---|---|---|
| MN-P52 | **P0** | *Neural Tangent Kernel: Convergence and Generalization in Neural Networks* (2018) | MN-03, MN-07, MN-16 | **Infinite-width neural networks** | 给出“lazy / kernel-like training”极限：训练可以主要改变 function，而不显著改变 representation；提供 feature-learning 的反面参照 | **Strong in its regime** |
| MN-P53 | **P0** | *Feature Learning in Infinite-Width Neural Networks* (2021) | MN-03, MN-07 | **Infinite-width networks / Word2Vec / MAML** | 明确指出 NTK limit 与 feature learning 是不同 regime；“模型是否真的学习 representation”本身是 training-dynamics 问题 | **Strong** |
| MN-P54 | **P0** | *Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks* (2014) | MN-03 | **Deep linear networks** | 深度本身就可以造成非线性的 learning dynamics、plateau、rapid transition；architecture 改变的不只是最终函数类，还包括学习路径 | **Strong in simplified setting** |
| MN-P55 | **P0** | *When Representations Align: Universality in Representation Learning Dynamics* (2024) | MN-03, MN-07, MN-09 | **多种 deep architectures** | 关键桥梁：不同 architecture 虽影响定量细节，但在足够灵活时可能共享某些 representation-learning dynamics；提示“architecture-specific mechanism”与“architecture-invariant dynamics”需要分开 | **Strong / high-value** |
| MN-P56 | **P0** | *Saddle-to-Saddle Dynamics Explains A Simplicity Bias Across Neural Network Architectures* (2026) | MN-03, MN-09 | **FC / ReLU / CNN / attention** | 给出跨 architecture 的统一训练动力学：模型沿 saddle/invariant manifold 逐步增加有效复杂度；简单性可能是 dynamics 产生的，而不只是最终参数分布 | **Strong, recent** |
| MN-P57 | **P0** | *Deep Double Descent: Where Bigger Models and More Data Hurt* (2020) | MN-03, MN-01, MN-02 | **CNN / ResNet / Transformer** | 破坏“更大模型一定更好、训练越久越差/越好”的简单叙事；model size、data size、training time 共同改变 effective complexity | **Strong** |
| MN-P58 | P1 | *Understanding Deep Learning Requires Rethinking Generalization* (2017) | MN-03, MN-16 | **CNN / shallow nets** | 说明 interpolation / memorization 能力与 generalization 不是简单二选一；capacity 不能直接解释模型为何学到有用结构 | **Strong** |
| MN-P59 | **P0** | *Prevalence of Neural Collapse During the Terminal Phase of Deep Learning Training* (2020) | MN-03, MN-07 | **VGG / ResNet-like classifiers** | 极端训练阶段会形成高度规则的 feature geometry；直接把 training dynamics 与 representation geometry 连起来 | **Strong for classification regime** |
| MN-P60 | P1 | *Abrupt and Spontaneous Strategy Switches Emerge in Simple Regularised Neural Networks* (2024) | MN-03, MN-09 | **Simple neural networks** | “突然出现的 strategy switch”可以由渐进 gradient learning + gating + noise 自然产生；行为上的突变不必对应训练过程本身的突变 | **Medium** |
| MN-P61 | P1 / recent | *How are Linear Representations Learned? Exact Solutions to the Dynamics of Abstraction* (2026) | MN-03, MN-07 | **Linear networks / nonlinear networks / DINOv3 / Gemma 4** | 把“概念方向如何形成”直接变成 training-dynamics 问题；data geometry、target geometry、depth、initialization、nonlinearity 都参与决定 abstraction | **Strong / recent** |

### 对 MN-03 的新认识

训练动力学现在越来越像一个真正的 **formation layer**：

\[
\boxed{
(\text{Architecture},\text{Data},\text{Objective},\text{Initialization})
\rightarrow
\text{Training Dynamics}
\rightarrow
\text{Representation / Circuit / Strategy}
}
\]

尤其要注意两个不同问题：

### ① “模型最终能表达什么？”

偏向：

\[
\text{Architecture / Expressivity}
\]

### ② “模型实际上会学成什么？”

偏向：

\[
\text{Training Dynamics / Implicit Bias}
\]

两者不能混为一谈。

---

# 4. Representation Geometry — 深化 MN-07

| ID | Priority | Paper | Node | Model family | 在图中的作用 | Evidence |
|---|---|---|---|---|---|---|
| MN-P62 | **P0** | *The Linear Representation Hypothesis and the Geometry of Large Language Models* (2024) | MN-07, MN-14 | **LLaMA-2** | 把“概念是 representation space 中的方向”形式化，并指出 geometry 的合理 inner product 可能不同于普通 Euclidean geometry | **Strong within formalization** |
| MN-P63 | **P0** | *The Geometry of Categorical and Hierarchical Concepts in Large Language Models* (2025) | MN-07, MN-15 | **Gemma / LLaMA-3** | 把 categorical / hierarchical concepts 与几何结构联系起来；representation geometry 可以承载层级语义，而非只有 binary directions | **Strong within setting** |
| MN-P64 | **P0** | *Toy Models of Superposition* (2022) | MN-07, MN-16 | **Small ReLU networks** | 一个表示维度可以通过 superposition 承载多个 feature；geometry 不等于“每个概念占一个干净维度”，存在 capacity–interference trade-off | **Strong in toy model** |
| MN-P65 | **P0** | *Towards Monosemanticity: Decomposing Language Models with Dictionary Learning* (2023) | MN-07 | **Small Transformers** | 从 neuron-centric view 转向 feature-centric view；representation 的基本分析单位未必是 neuron，而可能是 distributed feature direction | **Strong for analyzed models** |
| MN-P66 | **P0** | *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet* (2024) | MN-07, MN-08 | **Claude 3 Sonnet** | 将 feature-centric geometry 扩展到 production-scale LM，并展示可解释 feature 可以跨语言、跨模态、跨抽象层次出现且可被因果操纵 | **Strong / but incomplete** |
| MN-P67 | **P0** | *Function Vectors in Large Language Models* (2024) | MN-07, MN-08, MN-04 | **Autoregressive Transformers** | 说明 representation 不仅可以编码“内容/概念”，还可以编码“函数/任务”；representation → computation 的边界因此被推高 | **Strong in tested ICL settings** |
| MN-P68 | **P0** | *The Geometry of Concepts: Sparse Autoencoder Feature Structure* (2025) | MN-07 | **LLM SAE features** | 从 atom / brain / galaxy 多尺度观察 feature geometry，提示 representation space 本身具有跨尺度结构，而非随机点云 | **Medium–Strong** |
| MN-P69 | **P0** | *Similarity of Neural Representations Revisited* (2019) | MN-07, MN-09 | **CNN / deep vision models** | 用 CKA 系统比较不同网络/层的 representation similarity；提供跨 architecture 比较 representation geometry 的实用坐标系 | **Strong as an analysis framework** |
| MN-P70 | **P0 / recent** | *How are Linear Representations Learned? Exact Solutions to the Dynamics of Abstraction* (2026) | MN-03, MN-07 | **Linear / nonlinear networks; DINOv3 / Gemma 4 evidence** | 把静态 geometry 与动态 formation 接起来：abstraction 本身可以被视为 representation geometry 的形成过程 | **Strong in theory; emerging in frontier models** |
| MN-P71 | **P0** | *Neural Collapse Under MSE Loss: Proximity to and Dynamics on the Unconstrained Features Model* (2021) | MN-03, MN-07 | **Deep classifiers / unconstrained-features model** | 把 neural collapse 从一个静态观察进一步推进到可分析的 dynamics / idealized model；帮助连接“训练如何产生几何结构”与“最终几何长什么样” | **Strong within classification theory** |

---

# 5. 新增的跨节点高价值桥梁

下面这些不是简单“某个节点多了一篇 paper”，而是开始改变节点之间关系的论文。

| ID | 关系 | Paper | 关键认识 |
|---|---|---|---|
| MN-P72 | **Architecture ↔ State / Memory** | *Transformers are RNNs* (2020) | attention 的某些变体可以被重写成 recurrent state update；“有没有 state”首先取决于你如何定义 state / computation substrate |
| MN-P73 | **Architecture ↔ State / Representation** | *Transformers are SSMs* (2024) | Transformer 与 SSM 可以由共同的 structured matrix / state-space duality 联系起来；model family 边界未必等于最基本的 computational distinction |
| MN-P74 | **Architecture ↔ Representation Geometry** | *When Representations Align* (2024) | 不同 architecture 可能产生 qualitatively similar representation-learning dynamics；不能把某种 geometry 轻易归因给单一 architecture |
| MN-P75 | **Training Dynamics ↔ Representation Geometry** | *Neural Tangent Kernel* (2018) + *Feature Learning in Infinite-Width Neural Networks* (2021) | 关键不是“模型有没有 representation”，而是训练过程中 representation 是否真的被塑造；lazy 与 feature-learning 是两个不同 regime |
| MN-P76 | **Training Dynamics ↔ Architecture** | *Saddle-to-Saddle Dynamics...* (2026) | simplicity bias 可以由跨 architecture 的共同动力学产生；架构差异可能体现在“复杂度单位”而不是基本学习规律 |
| MN-P77 | **Geometry ↔ Memory** | *Understanding and Controlling the Geometry of Memory Organization in RNNs* (2025) | temporal memory 可以表现为 representation phase-space geometry 的组织变化 |
| MN-P78 | **Geometry ↔ Memory ↔ Knowledge** | *Attractor Geometry of Transformer Memory* (2026) | memory retrieval / parametric knowledge conflict 可以被视为 hidden-state geometry 中不同 attractor 的竞争 |
| MN-P79 | **Geometry ↔ Function / ICL** | *Function Vectors in Large Language Models* (2024) | representation 可以承载“要执行的函数”，因此 geometry 不只是 semantic content storage |
| MN-P80 | **Architecture ↔ Capability** | *Mamba-3: Improved Sequence Modeling using State Space Principles* (2026) | architecture 对 state tracking / retrieval 等能力存在可操作的结构性影响；architecture selection 可以直接围绕 capability requirement 做 |
| MN-P81 | **Data ↔ Architecture ↔ Dynamics** | *Understanding Unimodal Bias in Multimodal Deep Linear Networks* (2024) | architecture 的 modality fusion position 与 data statistics 会共同决定模型是否长期偏向某一 modality |
| MN-P82 | **Interface ↔ Representation** | *Tokenization Matters!* (2024) + *How Tokenization Limits Phonological Knowledge Representation...* (2026) | 输入接口本身会改变可学习 / 可表达的 representation，不应把 tokenizer 当作模型外部的纯工程细节 |
| MN-P83 | **Data ↔ Training ↔ Capability** | *DoReMi* (2023) + *DataComp-LM* (2024) | data mixture / filtering 本身可以显著改变训练效率和 downstream behavior；“模型能力”不能只归因于参数规模 |
| MN-P84 | **Scale ↔ Capability interpretation** | *Emergent Abilities of Large Language Models* (2022) + *Are Emergent Abilities of Large Language Models a Mirage?* (2023) | capability 的“突变”可能来自真实内部变化，也可能来自 metric；观察到 emergence 后需要先 audit measurement |
| MN-P85 | **Multimodal Representation** | *ImageBind: One Embedding Space To Bind Them All* (2023) | 不同 modality 可以通过共享 representation space 被绑定；“缺信息”有时可以通过改变 representation substrate / modality supply 来解决 |

---

# 6. 三个关键 Hub 的进一步认识

## 6.1 MN-09 Architecture：从“模型结构”升级为“substrate hypothesis”

当前最值得保留的高层认识不是：

> CNN / Transformer / SSM 哪个更好。

而是：

\[
\boxed{
\text{Architecture}
=
\text{a hypothesis about what computation should be natural}
}
\]

可以进一步拆成：

```text
Architecture
│
├── 1. Expressivity
│      └── 哪些函数/关系原则上容易或困难？
│
├── 2. Inductive Bias
│      └── 哪些结构会被优先偏好？
│
├── 3. Information Access
│      └── 信息通过什么路径被读取？
│
├── 4. State Update
│      └── 历史信息如何进入下一时刻？
│
├── 5. Computational Primitive
│      └── attention / recurrence / convolution / routing ...
│
└── 6. Representation Geometry
       └── 上述约束最终如何体现在内部空间？
```

### 最重要的关系变化

过去容易想：

\[
Architecture \rightarrow Capability
\]

现在更合理的是：

\[
\boxed{
Architecture
\rightarrow
\{\text{Expressivity, Access, State, Geometry, Dynamics}\}
\rightarrow
Capability
}
\]

这正是为什么跨 model-family 阅读对于当前项目有价值：

> **换 architecture 不只是换一个 backbone，而可能是在改变问题的“自然计算坐标系”。**

---

# 7.2 MN-03 Training Dynamics：从“优化过程”升级为“内部结构形成过程”

目前越来越明显：

\[
\boxed{
Training
\neq
\text{仅仅寻找一个低-loss parameter setting}
}
\]

而更接近：

\[
\boxed{
Training
=
\text{trajectory through a space of representations / circuits / strategies}
}
\]

其中至少存在：

```text
training
│
├── optimization regime
│     ├── lazy / kernel-like
│     └── feature learning
│
├── complexity recruitment
│     └── simplicity → more complex solution
│
├── representation formation
│     └── geometry / abstraction emerges
│
├── circuit formation
│     └── algorithmic mechanism becomes explicit
│
└── strategy switching
      └── behavior can change abruptly
```

所以如果未来看到：

> “这个模型有某种 representation。”

应该继续追问：

> **这是 architecture 给它的？data 给它的？objective 给它的？还是 training trajectory 自己形成的？**

这会直接影响我们如何设计 analysis experiment。

---

# 7.3 MN-07 Representation Geometry：从“可视化 hidden state”升级为内部组织语言

现在最值得保留的不是：

> “某个 concept 可以被线性 probe。”

而是：

\[
\boxed{
Representation\ Geometry
\text{ may be the common language connecting}
\begin{cases}
\text{concept}\\
\text{feature}\\
\text{state}\\
\text{memory}\\
\text{knowledge}\\
\text{function}\\
\text{world structure}
\end{cases}
}
\]

但这个认识必须拆成至少三个层次：

### Level 1 — Geometry exists

例如：

\[
\text{concept}\approx\text{direction / subspace / manifold}
\]

### Level 2 — Geometry has structure

例如：

\[
\text{hierarchy}
,\quad
\text{superposition}
,\quad
\text{clusters}
,\quad
\text{attractors}
,\quad
\text{alignment}
\]

### Level 3 — Geometry participates in computation

例如：

\[
\boxed{
\text{representation}
\rightarrow
\text{circuit}
\rightarrow
\text{behavior}
}
\]

Function Vectors、SAE steering、causal intervention 等开始提供 Level 3 的证据。

因此：

\[
\boxed{
\text{Probe evidence}
\rightarrow
\text{Geometric structure}
\rightarrow
\text{Causal participation}
}
\]

是比单纯“probe 出一个 feature”更完整的认识路径。

---

# 8. 三个 Hub 之间最值得关注的双向关系

## R-A｜Architecture → Representation Geometry

经典方向：

\[
\text{Architecture}
\rightarrow
\text{Inductive Bias}
\rightarrow
\text{Geometry}
\]

例如 convolutional locality、tree structure、SSM recurrence 等。

但不能把它理解成单向决定：

\[
\text{Geometry}
\not\equiv
f(\text{Architecture only})
\]

因为 data / objective / dynamics 也参与。

---

## R-B｜Training Dynamics → Representation Geometry

现在这是非常强的一条：

\[
\boxed{
\text{Training trajectory}
\rightarrow
\text{representation formation}
}
\]

NTK / feature-learning、Saxe、grokking、neural collapse、abstraction dynamics 都在不同尺度支持它。

这意味着：

> **representation geometry 是“结果”，training dynamics 是“形成过程”。**

---

## R-C｜Architecture → Training Dynamics

不是：

\[
Architecture\rightarrow Final\ Representation
\]

这么简单。

更合理：

\[
\boxed{
Architecture
\rightarrow
\text{optimization geometry / available modes}
\rightarrow
Training\ Dynamics
\rightarrow
Representation
}
\]

例如 saddle-to-saddle work 直接把 FC、CNN、attention 放进共同 dynamics 框架；Mamba/Transformer 对比则说明不同 state-update substrate 会改变 sequential learning 的自然路径。

---

## R-D｜Representation Geometry ↔ State / Memory

这是目前与当前 Video-MLLM 问题最直接的桥：

\[
\boxed{
\text{State}
\text{ may be better understood as a temporally organized structure in representation space}
}
\]

于是“state 是否存在”的问题可以进一步拆成：

```text
State
│
├── Is information present?
│       ↓
├── Is it geometrically organized?
│       ↓
├── Is it persistent / updated across time?
│       ↓
├── Can downstream computation access it?
│       ↓
└── Does it causally participate in the answer?
```

这比把 state 当成一个单独 hidden vector 更一般。

---

## R-E｜Representation Geometry ↔ Circuits

两者暂时不应该强行排成 parent → child。

更像：

\[
\boxed{
\text{Geometry} \Longleftrightarrow \text{Computation}
}
\]

因为：

- geometry 描述“信息组织在哪里、以什么结构存在”；
- circuit 描述“这些结构如何被读取、组合、写入”。

因此一个完整的 analysis 往往需要：

\[
\text{Where / What}
\rightarrow
\text{How organized}
\rightarrow
\text{How used}
\]

---

## R-F｜Representation Geometry → World / Latent Semantics

这里仍然是开放的：

\[
\text{structured representation}
\not\Rightarrow
\text{world model}
\]

需要额外证明：

\[
\text{entities}
+
\text{relations}
+
\text{state}
+
\text{dynamics}
+
\text{counterfactual consistency}
\]

是否被组织成一个可用于后续计算的 latent structure。

所以 MN-15 目前仍应保持为：

\[
\boxed{\text{OPEN}}
\]

而不是 geometry 的必然 downstream。

---

# 9. 一个新的“认识依赖”视角

当前最值得继续验证的不是所有 pairwise arrows，而是下面四条链：

### Chain A — Substrate

\[
\boxed{
Interface
\rightarrow
Architecture
\rightarrow
Expressivity / Access / State\ Mechanism
}
\]

回答：

> **这个模型天然能以什么方式处理信息？**

---

### Chain B — Formation

\[
\boxed{
Data + Objective + Scale + Architecture
\rightarrow
Training\ Dynamics
\rightarrow
Representation\ Geometry
}
\]

回答：

> **它为什么最终形成了这样的内部结构？**

---

### Chain C — Internal organization

\[
\boxed{
Representation\ Geometry
\Longleftrightarrow
Memory / Knowledge / Circuits
\rightarrow
Runtime\ Use
}
\]

回答：

> **模型内部到底如何组织并调用这些东西？**

---

### Chain D — Higher-order semantics

\[
\boxed{
Geometry + Memory + Knowledge + Computation
\rightarrow
World / Latent\ Semantics
\rightarrow
Capability
}
\]

回答：

> **这些局部结构是否足以组成更高层的“世界/任务结构”？**

---

# 10. 当前 P0 阅读队列

如果不是为了马上做某一个项目，而是为了形成长期的 model-nature 判断力，我建议下一轮优先不是继续平铺 paper list，而是沿着认识依赖阅读：

## Tier 1 — 三个 Hub 的骨架

1. **MN-P01** — *Geometric Deep Learning*
2. **MN-P55** — *When Representations Align*
3. **MN-P53** — *Feature Learning in Infinite-Width Neural Networks*
4. **MN-P56** — *Saddle-to-Saddle Dynamics...*
5. **MN-P62** — *Linear Representation Hypothesis and Geometry of LLMs*

这五篇可以先回答：

\[
\boxed{
Architecture
\rightarrow
Dynamics
\rightarrow
Geometry
}
\]

到底哪些关系是稳的。

---

## Tier 2 — Geometry → Memory / Computation

6. **MN-P42** — RNN memory geometry  
7. **MN-P41** — Transformer memory attractor geometry  
8. **MN-P20** — Transformer Circuits  
9. **MN-P67** — Function Vectors  
10. **MN-P64** — Toy Models of Superposition  
11. **MN-P66** — Scaling Monosemanticity  

目标：

\[
\boxed{
Geometry
\rightarrow
Memory / Feature / Function
\rightarrow
Causal\ Use
}
\]

---

## Tier 3 — Architecture as substrate selection

12. **MN-P46** — Transformers are SSMs  
13. **MN-P06** — Mamba  
14. **MN-P80** — Mamba-3  
15. **MN-P08** — Attention vs SSM ICL  
16. **MN-P47** — Syntax / tree inductive bias  
17. **MN-P48** — CNN inductive bias  

目标：

> **什么时候“换模型”不是 engineering choice，而是在换一个更适合某种 computation / representation 的 substrate？**

---

## Tier 4 — Formation / capability

18. **MN-P14 / P15** — Grokking  
19. **MN-P57** — Double Descent  
20. **MN-P59** — Neural Collapse  
21. **MN-P61** — Abstraction Dynamics  
22. **MN-P84** — Emergence vs metric artifact  
23. **MN-P83** — Data mixture / curation  

目标：

\[
\boxed{
为什么能力、representation、strategy 会在训练过程中以这种方式出现？
}
\]

---

# 11. 下一轮真正值得深挖的问题

当前最值得继续做的已经不是：

> “还有哪些 representation geometry paper？”

而是以下五个问题：

### Q1. Architecture 到底通过什么中介影响 Representation？

候选：

\[
Architecture
\rightarrow
\begin{cases}
\text{symmetry}\\
\text{locality}\\
\text{state update}\\
\text{access pattern}\\
\text{optimization geometry}
\end{cases}
\rightarrow
Representation
\]

---

### Q2. Training Dynamics 是否存在跨 architecture 的共同规律？

尤其验证：

\[
\boxed{
\text{architecture-specific mechanism}
\quad vs \quad
\text{architecture-invariant dynamics}
}
\]

`When Representations Align` 与 `Saddle-to-Saddle Dynamics` 是这一问题的两个关键入口。

---

### Q3. Representation Geometry 到底什么时候具有“memory / state”语义？

需要区分：

\[
\text{information present}
\]

\[
\text{information persistent}
\]

\[
\text{information organized}
\]

\[
\text{information accessible}
\]

\[
\text{information causally used}
\]

这会直接连接到当前 Video-MLLM 的 state-utilization 问题。

---

### Q4. Geometry 与 Circuit 的关系究竟是什么？

候选有三种：

```text
A. Geometry → Circuit
B. Circuit → Geometry
C. Geometry ↔ Circuit
```

目前证据更支持 **C**，但需要跨 Transformer / CNN / RNN / SSM / multimodal architectures 继续验证。

---

### Q5. 什么证据足以从“structured representation”升级到“world model”？

这是 MN-15 当前最重要的开放边界：

\[
\boxed{
Representation
\rightarrow
State
\rightarrow
Dynamics
\rightarrow
Counterfactual\ Use
\rightarrow
World\ Model
}
\]

目前不应该直接把其中任意一个中间结果叫作 world model。

---

# 12. 当前暂存结论

这一轮之后，Model-Nature 图里最值得保留的不是“16 个方向”，而是三个层级：

\[
\boxed{
\textbf{SUBSTRATE}
\rightarrow
\textbf{FORMATION}
\rightarrow
\textbf{INTERNAL ORGANIZATION}
}
\]

分别由：

\[
\boxed{
Architecture
}
\qquad
\boxed{
Training\ Dynamics
}
\qquad
\boxed{
Representation\ Geometry
}
\]

作为三个主要 hub。

然后：

\[
\boxed{
Memory,\ Knowledge,\ Circuits,\ Access,\ ICL
}
\]

更多表现为内部组织 / runtime 的不同侧面；

而：

\[
\boxed{
World / Latent\ Semantics
}
\]

暂时应被视为更高阶、尚未闭合的 hypothesis；

最后：

\[
\boxed{
Capability / Limit
}
\]

才是最常被观察到的 downstream phenomenon。

> **下一阶段不应继续无结构地增加 paper 数量。**
>
> 应该开始对这三个 hub 分而治之：每个 hub 都建立自己的 **canonical literature → competing interpretations → cross-family evidence → boundary conditions → downstream consequences**，然后重新检查 hub 之间的箭头是否真的成立。
