# High\-level 认识维度与相互关系

\[Model\-Nature\_High\-Level\_Recognition\_Paper\_List\_v1\_1\.md\]

**簇关系图**



$\underbrace{\text{Interface + Architecture}}_{\text{substrate}} + \underbrace{\text{Data + Objective + Scale}}_{\text{formation conditions}} \rightarrow \underbrace{\text{Training Dynamics}}_{\text{formation process}} \rightarrow \underbrace{\text{representation / memory / computation / access}}_{\text{Internal Organization}} \rightarrow \underbrace{\text{Runtime Computation}}_{\text{adaptation}} \rightarrow \underbrace{\text{Capability}}_{\text{Downstream phenomenon}}  $

其中 **Expressivity** 是贯穿其中的 constraint，**World Model** 是多个内部结构组合以后可能形成的 higher\-order hypothesis。

# 分别理解

## MN\-01｜Scaling：能力本身有一个“资源轴”

Scaling Laws 以及 Chinchilla 都提供了一个非常强的认识：

$\text{Capability} \not\approx \text{Architecture alone}$

而与：

$\text{Model Size} ,\quad \text{Data} ,\quad \text{Compute}$

存在可预测的关系。Kaplan 等发现 language\-model loss 随模型规模、数据量和 compute 呈 power\-law；Chinchilla 又进一步显示，在固定训练计算预算下，model size 与 training tokens 的配比本身就是重要变量。[arXiv](https://arxiv.org/abs/2001.08361?utm_source=chatgpt.com)

**以后值得问：**

> 一个我们观察到的“机制差异”，到底是 architecture effect，还是 scale effect？
> 
> 

这会直接影响很多实验的 interpretation。

---

## MN\-02｜Data：训练数据不是背景，而是 model substrate 的一部分

例如数据去重研究发现，训练语料中的重复和 train\-test overlap 会直接改变 memorization、训练效率和 evaluation。[ACL Anthology](https://aclanthology.org/2022.acl-long.577/?utm_source=chatgpt.com)

所以：

$\text{Model} \neq \text{Architecture + Parameters}$

更接近：

$\text{Model} = \text{Architecture} + \text{Optimization} + \text{Training Distribution}$

**以后值得问：**

> 当某种 capability / representation 出现时，它究竟是 architecture naturally produces 的，还是 training distribution 强烈诱导出来的？
> 
> 

---

## MN\-03｜Training dynamics：模型内部的“认识”可能不是静态形成的

Grokking 提供了一个特别重要的认识：

> training loss 已经表现出 overfitting 后，模型内部仍可能继续发生结构性变化，最终突然进入更强的 generalizing solution。[arXiv](https://arxiv.org/abs/2201.02177?utm_source=chatgpt.com)
> 
> 

因此：

$\text{Training} \neq \text{monotonic accumulation of knowledge}$

可能存在：

$\text{memorization} \rightarrow \text{representation reorganization} \rightarrow \text{algorithmic/generalizing solution}$

**以后值得问：**

> 一个 frontier model 当前呈现的 representation / circuit，是“从一开始就存在”，还是训练后期发生了结构跃迁？
> 
> 

---

## MN\-04｜In\-context learning：模型能够在 inference 时进行某种“学习”

Induction\-head 工作把 ICL 与特定 circuit 联系起来；另一条工作甚至在简单 regression setting 中展示了 Transformer forward pass 可以近似执行 gradient descent / meta\-learning。[arXiv](https://arxiv.org/abs/2209.11895?utm_source=chatgpt.com)

所以一个非常重要的高层认识是：

$\text{Pretrained Model} + \text{Context} \rightarrow \text{Temporary Adaptation}$

这意味着：

> **模型的“学习”不一定发生在参数更新阶段。**
> 
> 

这和 state/memory 是强邻接，但不是同一个问题。

---

## MN\-05｜State / memory topology：历史信息到底存在哪里、怎么流动

这是你当前最熟悉的一个，但现在应该把它放回更大的地图。

不同 architecture 给出了完全不同的答案：

- recurrent model 有显式递归 state；

- Transformer 通过 token representations / attention 访问 context；

- Mamba 使用 selective state；

- Transformer/SSM 之间甚至可以在统一的 state\-space duality 下重新理解。[DOI](https://doi.org/10.48550/arXiv.2312.00752?utm_source=chatgpt.com)

所以值得调查的并不是：

> “LLM 有没有 state？”
> 
> 

而是更大的：

$\boxed{ \text{What is the memory topology of a model?} }$

包括：

- write；

- retain；

- compress；

- retrieve；

- overwrite；

- forget；

- route；

- query\-dependent reconstruction。

这才是后续 state track 的真正上层母问题。

---

## MN\-06｜Knowledge ≠ Output：模型内部“知道什么”与“说什么”可能是两个对象

LAMA 很早就发现 pretrained language models 可以包含 relational knowledge；ROME 又通过 causal intervention 找到并编辑 factual associations；latent\-knowledge 工作则进一步明确研究：

> **what models know vs what models say**。[ACL Anthology](https://aclanthology.org/D19-1250/?utm_source=chatgpt.com)
> 
> 

于是：

$\text{Internal Knowledge} \neq \text{Output Behavior}$

这个 distinction 对解释实验尤其重要。

一个 failure：

> 模型回答错了
> 
> 

不能直接推出：

> 模型内部没有这个知识。
> 
> 

反过来：

> probe 能读出知识
> 
> 

也不能直接推出：

> 模型会在 task 中调用它。
> 
> 

---

## MN\-07｜Representation geometry：representation 可能不是“一个地方存一个东西”

Superposition 工作给出了一个重要的机制性认识：有限维 activation space 中可以叠加大量 sparse features，因此一个 neuron / dimension 并不一定对应一个语义对象。[Hugging Face](https://huggingface.co/papers/2209.10652?utm_source=chatgpt.com)

另一方面，Platonic Representation Hypothesis 提出，不同模型/模态的 representation 可能随着规模增加出现某种收敛趋势。[Hugging Face](https://huggingface.co/papers/2405.07987?utm_source=chatgpt.com)

所以这里至少存在两个需要调查的高层问题：

$\text{Representation} = ?$

究竟更接近：

- localized features；

- distributed features；

- superposed features；

- geometric directions；

- shared latent structure；

- model\-specific basis。

这会直接决定以后：

> probe / patch / tracing 找到的到底能不能被称为“representation”。
> 
> 

---

## MN\-08｜Computation as circuits：能力可能来自可组合的计算结构

Transformer Circuits 的核心认识不是：

> “某个 neuron 表示某个概念。”
> 
> 

而是：

> **模型由多个相互通信、组合的 computational components 构成。**
> 
> 

Residual stream 被视为 communication channel，attention head 可以分解为 QK 与 OV circuits，head 之间还能发生 composition。

Induction heads 又给出了一个比较干净的例子：

$\text{capability} \rightarrow \text{specific computational mechanism}$

因此未来可以独立研究：

> capability 是不是应该被理解成 **distributed computation**，而不是一个 representation object。
> 
> 

---

## MN\-09｜Architecture → Inductive Bias → Expressivity Limit

这是我认为你特别应该加入长期 Model\-nature reading 的一大类。

例如 GNN 的理论工作直接证明 message\-passing architecture 存在明确的 graph distinguishability limit；Deep Sets 则把 permutation invariance 直接放入 architecture 的函数形式。[arXiv](https://arxiv.org/abs/1810.00826?utm_source=chatgpt.com)

所以：

$\text{Failure}$

至少存在两个完全不同的解释：

$\text{model didn't learn it}$

vs\.

$\boxed{\text{model cannot represent it}}$

后者一旦成立，研究方向就完全不同了：

$\text{Training more} \rightarrow \text{No}$

可能应该：

$\text{Change substrate}$

这就是你说的 diffusion / architecture\-nature 文献为什么重要。

---

## MN\-10｜Conditional computation / modularity

Switch Transformer 等 MoE 工作提供另一种 model\-nature：

$\text{Input} \rightarrow \text{Routing} \rightarrow \text{Subset of Parameters}$

也就是说，大模型并不一定是：

> 所有输入都经过同一个统一 computation。
> 
> 

而可能是：

> **不同输入激活不同部分的 model capacity。**
> 
> 

Switch Transformer 通过 sparse expert routing 实现了大量参数而保持相对固定的计算成本。[arXiv](https://arxiv.org/abs/2101.03961?utm_source=chatgpt.com)

以后值得调查：

> specialization 到底发生在 representation、parameters、circuits 还是 routing level？
> 
> 

---

## MN\-11｜Test\-time compute：模型能力还有一个 inference\-time 轴

CoT 已经显示，额外的 intermediate computation 可以显著提高复杂 reasoning；后续工作进一步把 test\-time compute 当成一个可以独立 scaling 的资源，并发现其收益依赖 problem difficulty。[arXiv](https://arxiv.org/abs/2201.11903?utm_source=chatgpt.com)

这带来一个重要认识：

$\text{Capability} = f( \text{pretraining compute}, \text{model scale}, \text{test-time compute} )$

所以某个模型：

> “能不能做 X”
> 
> 

实际上可能不是静态能力，而是：

> **在给定 inference budget 下能不能做 X。**
> 
> 

---

## MN\-12｜Context length ≠ Context utilization

Lost in the Middle 是非常典型的高层认识：

> 模型可以接受很长 context，并不意味着它能均匀、可靠地访问其中的信息。
> 
> 

在 controlled position experiments 中，相关信息放在长 context 中间时性能会明显下降。[DOI](https://doi.org/10.48550/arXiv.2307.03172?utm_source=chatgpt.com)

因此：

$\text{Context Capacity} \neq \text{Context Accessibility} \neq \text{Context Utilization}$

这其实与 MN\-05 state/memory、MN\-04 ICL、MN\-11 test\-time computation 都存在交叉，但值得独立成为一个认识领域。

---

## MN\-13｜Tokenization / interface granularity

Tokenization 很容易被当成工程细节，但理论工作表明，它可能直接影响 Transformer 能否有效学习某些 sequence distributions。[arXiv](https://arxiv.org/abs/2404.08335?utm_source=chatgpt.com)

所以：

$\text{Representation difficulty}$

可能不是从 hidden layer 才开始的，而可能早在：

$\boxed{\text{input discretization}}$

就已经被决定了一部分。

这也是一种典型的：

> **interface itself is inductive bias**
> 
> 

的认识。

---

## MN\-14｜Objective / post\-training：模型最终“是什么”取决于训练阶段

InstructGPT 展示了 human\-feedback post\-training 可以显著改变行为，即使基础模型规模更大；Tulu 3 则进一步显示 SFT、DPO、RLVR 等不同 post\-training recipe 对能力和行为有实质影响。[arXiv](https://arxiv.org/abs/2203.02155?utm_source=chatgpt.com)

更重要的是 reward overoptimization 工作表明：

$\text{optimized proxy} \neq \text{true objective}$

甚至存在：

$\text{reward} \uparrow \qquad \text{true quality} \downarrow$

的情况。[arXiv](https://arxiv.org/abs/2210.10760?utm_source=chatgpt.com)

所以未来需要区分：

> pretrained model 的 latent capability
> 
> 

和：

> post\-trained model 的 behavioral policy。
> 
> 

---

## MN\-15｜World model / latent semantics

这是我会放进地图、但**暂时不把它当事实**的一类。

核心问题：

$\boxed{ \text{Does next-token prediction induce an internal model of the world?} }$

也就是：

> 模型是否只是在学习语言分布，还是会形成关于人物、事件、因果、物理状态等 latent structure？
> 
> 

这个方向已经有大量讨论，例如 Jacob Andreas 关于 language models as world models 的工作/报告。[CBMM](https://cbmm.mit.edu/video/language-models-world-models?utm_source=chatgpt.com)

但它与 MN\-05/MN\-07/MN\-08 不同：

- state 是“怎么保存/更新”；

- representation 是“怎么编码”；

- circuit 是“怎么计算”；

- world model 是“**内部究竟是否形成了某种关于外部世界的结构性模型**”。

所以它应该作为一个**开放假设**进入后续调查，而不是当前 framework 的前提。





# 依赖关系理解

**依赖关系图**

```Plain Text
┌──────────────────────┐
                         │ Tokenization /       │
                         │ Interface Granularity│
                         └──────────┬───────────┘
                                    ↓
                         ┌──────────────────────┐
                         │ Architecture /       │
                         │ Inductive Bias       │
                         └──────┬─────┬─────────┘
                                │     │
                    ┌───────────┘     └───────────────┐
                    ↓                                 ↓
        ┌─────────────────────┐          ┌─────────────────────┐
        │ Expressivity /      │          │ Representation      │
        │ Equivalence Classes │          │ Geometry             │
        └──────────┬──────────┘          └──────────┬──────────┘
                   │                                │
                   │                                │
                   │                    ┌───────────┼───────────┐
                   │                    ↓           ↓           ↓
                   │              ┌──────────┐ ┌──────────┐ ┌──────────┐
                   │              │ State /  │ │Knowledge │ │ Circuits │
                   │              │ Memory   │ │          │ │ / Comp.  │
                   │              │ Topology │ │          │ │          │
                   │              └────┬─────┘ └────┬─────┘ └────┬─────┘
                   │                   │            │            │
                   │                   └──────┬─────┴─────┬──────┘
                   │                          ↓           ↓
                   │                 ┌────────────────────────┐
                   │                 │ Context Access /       │
                   │                 │ Retrieval               │
                   │                 └───────────┬────────────┘
                   │                             ↓
                   │                    ┌─────────────────┐
                   │                    │ ICL / Runtime   │
                   │                    │ Adaptation      │
                   │                    └────────┬────────┘
                   │                             ↓
                   │                    ┌─────────────────┐
                   │                    │ Test-Time       │
                   │                    │ Computation     │
                   │                    └────────┬────────┘
                   │                             ↓
                   └──────────────────────→ Capability
```

与此同时，从另一侧有：

```Plain Text
Data
   │
Objective / Post-training
   │
Scale
   │
   └───────┐
           ↓
    Training Dynamics
           ↓
   ┌───────┴────────────────────┐
   ↓                            ↓
Representation             Circuits
   ↓                            ↓
State / Memory              Knowledge
   ↓                            ↓
Context Access            World / Latent Semantics
   └──────────────┬─────────────┘
                  ↓
             Capability
```

而 **Conditional Computation** 横跨：

```Plain Text
Architecture
      ↕
Conditional Routing
      ↕
Representation / Circuits
      ↕
Runtime Computation
```

- **M\-arrow（mechanistic）**：改变 A 是否会改变 B；

- **E\-arrow（epistemic）**：认识 A 是否显著改变我们理解 B 的方式。

这一区分很重要。你提出的：

$\text{Representation Geometry}\rightarrow\text{State}$

我认为主要首先是一个 **E\-arrow**，其次才是 M\-arrow。

---

## A01｜Tokenization / Interface → Representation / Expressivity

**判断：Strong**

Tokenization 并不是纯 preprocessing，而会改变模型首先接收到的离散对象，因此可以影响哪些结构容易表示和学习。

例如针对 LLM 的 adversarial tokenization 工作显示，改变 token segmentation 就能显著破坏 GPT\-4o、Llama\-3、Qwen2\.5\-max 等模型的能力；2026 ACL 的工作进一步直接把 tokenization mismatch 与 phonological representation 的下降联系起来。[arXiv](https://arxiv.org/abs/2405.17067?utm_source=chatgpt.com)

所以：

$\boxed{ \text{Interface} \rightarrow \text{Representational Possibility} }$

是可以保留的。

---

## A02｜Architecture → Expressivity

**判断：Very Strong**

这是目前整个图里最稳的箭头之一。

Deep Sets 从函数形式直接刻画 permutation\-invariant / equivariant structure；GNN 工作则直接证明 message\-passing architecture 对某些 graph structures 无法区分；Transformer 理论工作则证明了其 universal approximation 条件。[arXiv](https://arxiv.org/abs/1703.06114?utm_source=chatgpt.com)

因此：

$\boxed{ Architecture \rightarrow Expressivity / Equivalence Classes }$

几乎可以作为一个 foundational relation。

---

## A03｜Architecture → Representation Geometry

**判断：Strong，但不是单向因果**

Geometric Deep Learning 本身就是从 architecture 的 symmetry / locality / equivariance 等 inductive bias 出发理解 representation。它把 CNN、RNN、GNN、Transformer 放进一个共同的几何框架。[arXiv](https://arxiv.org/abs/2104.13478?utm_source=chatgpt.com)

ConViT 又提供了很好的具体例子：给 ViT 注入 convolutional locality bias 后，模型的学习行为和 sample efficiency 会发生系统性变化。[arXiv](https://arxiv.org/abs/2103.10697?utm_source=chatgpt.com)

所以：

$Architecture \rightarrow Representation\ Geometry$

成立。

但不能理解成：

> architecture 完全决定 geometry。
> 
> 

数据、objective、scale、training dynamics 也会塑造 geometry。

因此它更像：

$\boxed{ Architecture \rightarrow \text{biases the space of possible geometries} }$

而不是唯一决定 geometry。

---

# 一个非常重要的中间节点：Training Dynamics

这里我觉得我们原来低估了它。

应该有：

$\boxed{ Data + Objective + Scale + Architecture \rightarrow Training\ Dynamics \rightarrow Internal\ Organization }$

而不是：

$Data \rightarrow Capability$

这种过于粗糙的关系。

---

## A04｜Data / Objective / Scale → Training Dynamics

**判断：Strong**

Scaling Laws 建立了 model size / data / compute 与训练 loss 的系统关系；Chinchilla 又证明 model/data allocation 本身会改变 compute\-optimal training。[arXiv](https://arxiv.org/abs/2001.08361?utm_source=chatgpt.com)

而 Grokking 与 Progress Measures 更进一步告诉我们：

> training 不是简单地“loss 一直下降”。
> 
> 

在受控 Transformer 中，可以观察到：

$\text{memorization} \rightarrow \text{circuit formation} \rightarrow \text{cleanup}$

这样的内部结构形成过程。[arXiv](https://arxiv.org/abs/2201.02177?utm_source=chatgpt.com)

因此 Training Dynamics 是一个真正的**桥梁节点**。

---

# Representation Geometry → State / Memory

这是你刚才提出的那个问题。

我现在的判断是：

$\boxed{ \textbf{这是一个非常重要的 epistemic upstream arrow。} }$

但不是简单的 causal parent。

---

## A05｜Representation Geometry → State / Memory

**E\-strength：Very Strong**

因为要问：

> “模型有没有维护 state？”
> 
> 

首先必须知道：

> “一个 state 在 representation space 中究竟是什么？”
> 
> 

例如它可能不是：

$h_t \approx h_{t+1}$

而可能是：

$\boxed{ \text{某个 task-relevant relation 在 representation geometry 中持续可恢复} }$

RNN 的 memory\-geometry 工作直接发现，working\-memory encoding 和 retention 会对应不同的 latent subspaces / transformations；2025 年另一项 RNN 工作更直接观察到 memory organization 的 phase\-space geometric restructuring。[arXiv](https://arxiv.org/abs/2411.02685?utm_source=chatgpt.com)

而 2026 年的 *Attractor Geometry of Transformer Memory* 已经把这一思想推进到 Transformer：它把 parametric memory 与 working memory 的冲突理解为 hidden\-state space 中 attractor basin 的竞争。[arXiv](https://arxiv.org/abs/2605.05686?utm_source=chatgpt.com)

更关键的是，最新的 Transformer–SSM 比较工作已经开始把：

- evidence assembly

- belief maintenance

- addressing

统一放进一个 representational **belief geometry** 中，并用 LLaMA\-type Transformer 与 Mamba\-2 做比较。[arXiv](https://arxiv.org/abs/2609.32341?utm_source=chatgpt.com)

所以现在这条关系已经不只是我的概念推测了。

---

# 但 Geometry 和 Circuits 不应该排成简单上下游

这是我对上一轮结构的一个重要修正。

## A06｜Representation Geometry ↔ Circuits / Computation

**判断：Coupled rather than simple arrow**

Transformer Circuits framework 中，residual stream 本身就是 communication channel，不同 components 从不同 subspaces 读取/写入信息；attention head 的 composition 又形成更复杂的 computational paths。[变压器电路](https://www.transformer-circuits.pub/2021/framework/index.html?utm_source=chatgpt.com)

因此：

$\text{Geometry} \rightarrow \text{Circuit}$

有一定意义，但反过来也成立：

$\text{Circuit} \rightarrow \text{Geometry}$

因为一个 circuit 反复读写某些 subspaces，本身就在塑造这些 representation。

所以这里最好标成：

$\boxed{ \text{Representation Geometry} \;\Longleftrightarrow\; \text{Computational Circuits} }$

而不是硬排：

> Geometry 是上游，Circuit 是下游。
> 
> 

这两个更像**同一个内部计算系统的“状态空间”和“运算过程”两个侧面**。

---

# State / Memory → Context Access，也需要降级一点

## A07｜State / Memory → Context Access / Retrieval

**判断：Medium**

State\-space / recurrent architectures 确实通过 state compression / propagation 解决长序列信息保留；Transformer\-XL 和 Compressive Transformer 则明确把 past information 作为 memory 处理。Mamba 又进一步允许 state update 根据输入选择性传播/遗忘。[arXiv](https://arxiv.org/abs/1911.05507?utm_source=chatgpt.com)

但是：

> **Context access 不一定需要一个显式 compressed state。**
> 
> 

标准 Transformer 可以直接通过 attention 对 context 进行访问。

所以正确关系更像：

$\boxed{ Memory\ Mechanism \rightarrow \text{one possible substrate for Access} }$

而不是：

$Memory \rightarrow Access$

必然成立。

最新 attention\-vs\-SSM 工作反而非常漂亮地说明了这一点：**belief maintenance** 和 **content addressing** 可以由不同 architecture 优势分别承担。[arXiv](https://arxiv.org/abs/2609.32341?utm_source=chatgpt.com)

---

# Context Access → ICL：比较强，但 Access 不是 ICL 的充分条件

## A08｜Context Access \+ Representation \+ Computation → ICL

**判断：Strong as a conjunctive relation**

ICL 不能简单写成：

$Access\rightarrow ICL$

因为至少还需要：

$\text{Representation} + \text{Access} + \text{Computation}$

Induction\-head 工作发现特定 attention\-head circuit 与 ICL 的出现高度同步，并在小模型中提供了较强的因果证据。[arXiv](https://arxiv.org/abs/2209.11895?utm_source=chatgpt.com)

Akyürek 等进一步发现 Transformer 可以在 activation 中隐式编码 context\-specific predictors；von Oswald 等则在受控 regression setting 中展示了 Transformer 可以在 forward pass 中实现类似 gradient descent 的计算。[arXiv](https://arxiv.org/abs/2211.15661?utm_source=chatgpt.com)

所以：

$\boxed{ \text{Context Access} + \text{Representation} + \text{Computational Mechanism} \rightarrow \text{ICL} }$

比单独的 `Access → ICL` 更准确。

---

# Parametric Knowledge 与 Context Memory：不是上下游，而是双源竞争/协同

## A09｜Parametric Knowledge ↔ Contextual Information

**判断：Coupled / competing sources**

这条关系非常明确。

Transformer 可以同时从：

$\text{weights}$

和：

$\text{current context}$

取得信息，而且两种 information source 的 generalization bias 不同。[arXiv](https://arxiv.org/abs/2210.05675?utm_source=chatgpt.com)

更近期的工作在多个 LLM 上直接研究 context 与 parametric memory 的分配；另一个对 Atlas 的 causal mediation 分析则进一步分解了“判断 context 是否 relevant”以及“如何利用 context”的内部机制。[arXiv](https://arxiv.org/abs/2409.08435?utm_source=chatgpt.com)

所以：

$\boxed{ \text{Parametric Memory} \;\Longleftrightarrow\; \text{Contextual Memory} }$

应该成为一个独立关系，而不是：

> Parametric Knowledge → Context Access。
> 
> 

---

# Circuits → ICL / Capability：这是典型的中介关系

## A10｜Circuits → Runtime Capability

**判断：Strong in controlled settings; medium for frontier models**

Induction heads 是非常漂亮的例子：

$\text{circuit} \rightarrow \text{algorithmic behavior} \rightarrow \text{ICL}$

而不是：

$\text{representation} \rightarrow \text{answer}$

Tuned Lens 又从另一个角度把 Transformer 理解成逐层 refinement 的 iterative inference process。[arXiv](https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com)

因此 Circuit 更像：

> **内部计算机制的中介层。**
> 
> 

---

# Conditional Computation 是一个“横跨层级”的节点

它不能简单塞进“模型如何计算”。

更准确的是：

$\boxed{ Architecture \leftrightarrow Routing \leftrightarrow Conditional\ Computation \leftrightarrow Circuits \leftrightarrow Capability }$

Switch Transformer 的核心就是：

$x \rightarrow \text{router} \rightarrow E_i$

即不同输入只激活部分 parameter capacity。

所以 Conditional Computation 是：

> **Architecture\-level inductive bias 与 runtime computation 之间的桥。**
> 
> 

这也是为什么它值得作为独立节点保留。

---

# Test\-Time Compute 是明显的 downstream/runtime mechanism

## A11｜Test\-Time Compute → Capability

**判断：Strong**

这一条不需要再放得太高。

它本质上是：

$\text{Runtime Compute} \rightarrow \text{More Search / Reasoning} \rightarrow \text{Better Capability}$

Snell 等的工作直接研究了 test\-time compute scaling，并发现收益高度依赖 prompt difficulty；在 FLOPs\-matched 条件下，小模型增加 test\-time compute 可以超过大得多的模型。[arXiv](https://arxiv.org/abs/2408.03314?utm_source=chatgpt.com)

所以：

> **Test\-Time Compute 是非常重要的 runtime lever，但不是理解“模型内部是什么”的最高层认识。**
> 
> 

---

# Post\-training / Objective → Behavior 很强，但 → Internal Geometry 要谨慎

InstructGPT 已经直接证明：

> 相同基础 GPT\-3，通过 SFT \+ RLHF 后，行为可以发生巨大变化，甚至小得多的模型能够在 human preference 上超过大得多的 base model。[arXiv](https://arxiv.org/abs/2203.02155?utm_source=chatgpt.com)
> 
> 

Process supervision 工作又显示改变 supervision granularity 可以显著改变 reasoning behavior。[arXiv](https://arxiv.org/abs/2305.20050?utm_source=chatgpt.com)

所以：

$\boxed{ Objective / Post-training \rightarrow Behavior }$

非常强。

但：

$Objective \rightarrow \text{specific internal geometry}$

目前证据没有同样强，因此暂时不要画得太实。

---

# World Model：我现在明确不把它当 downstream

这是一个非常重要的分类修正。

如果：

$\text{Capability}$

是：

> 模型能不能做出来。
> 
> 

那么：

$\text{World Model}$

是在问：

> **模型内部是否存在一个能够组织实体、关系、状态、空间、时间、规则等东西的更高层结构？**
> 
> 

所以它更像：

$\boxed{ Representation + Memory + Knowledge + Computation \rightarrow World\ Model\ Hypothesis }$

而不是：

$Capability \rightarrow World\ Model$

---

## A12｜Internal Organization → World / Latent Semantics

**判断：Open / Medium**

Othello\-GPT 是一个非常干净的 domain\-specific case：Transformer 仅训练 next\-token prediction，却形成了可解码、可干预的 board\-state representation。[arXiv](https://arxiv.org/abs/2210.13382?utm_source=chatgpt.com)

*Language Models Represent Space and Time* 又在 Llama\-2 / Pythia 中发现跨尺度的空间、时间线性表示。[arXiv](https://arxiv.org/abs/2310.02207?utm_source=chatgpt.com)

*Implicit Representations of Meaning* 则在 BART/T5 中发现动态的 entity/situation representations，可以随 discourse 状态变化。[arXiv](https://arxiv.org/abs/2106.00737?utm_source=chatgpt.com)

但这里一定要保留一个巨大 caveat：

> **这些结果支持“存在结构化 latent representations”，还不足以证明自然语言 LLM 拥有完整 dynamic causal world model。**
> 
> 

*From task structures to world models* 自己就把 Othello 这种 domain\-specific evidence 与 natural\-language LLM 的 generalization 区分开。[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1364661324000354?utm_source=chatgpt.com)

所以这是目前整张图中最应该标成：

$\boxed{\text{OPEN}}$

的一个高层认识。







## Chain 1

$\boxed{ \text{Representation} \quad \text{Memory} \quad \text{Computation} \quad \text{Access} }$

它们分别回答四个不同的问题。

这四个其实不是四个孤立研究方向，而更像是一个内部计算系统的四个互补维度。

---

### 1\.1 Representation：What is encoded?

例如：

$x_t \rightarrow z_t$

我们首先要知道：

> 红球、蓝球、身份、位置、因果关系这些东西在 $z_t$ 中究竟以什么结构存在？
> 
> 

这里就是 Representation Geometry。

---

### 1\.2 Memory：What survives?

然后才问：

$z_t \rightarrow z_{t+1}$

发生了什么？

- 哪些信息被保留？

- 哪些被遗忘？

- 哪些被压缩？

- 哪些被覆盖？

- 哪些信息可以跨很长时间保留？

这就是 State / Memory Topology。

Transformer\-XL 通过 segment\-level recurrence 明确改变了这种 temporal information path；Mamba 则把 state update 本身做成 input\-dependent 的 selective propagation / forgetting。[arXiv](https://arxiv.org/abs/1901.02860?utm_source=chatgpt.com)

所以你会发现：

> **Memory topology 本质上是在研究 representation 在时间维度上的动力学。**
> 
> 

这就是为什么 Representation Geometry 会天然帮助 State。

---

### 1\.3 Computation：What happens to it?

即使信息存在、也被保存下来，还要问：

> 模型拿到这个 representation 之后做了什么？
> 
> 

例如：

$z_{\text{object}} \rightarrow z_{\text{relation}} \rightarrow z_{\text{answer}}$

这就是 Circuits / Composition。

Transformer Circuits 的经典表述其实非常适合说明这一点：residual stream 可以被理解成通信通道，而 attention heads 的核心功能之一就是在 token positions 之间移动信息；不同 head 还可以发生 composition。[变压器电路](https://transformer-circuits.pub/2021/framework?utm_source=chatgpt.com)

因此：

> **Representation 是“东西”，Circuit 是“对东西做什么”。**
> 
> 

---

### 1\.4 Access：Which information gets used?

然后还有一个容易和前三者混淆的问题：

> 信息虽然存在，但模型有没有把它找出来？
> 
> 

这就是 Context Access / Retrieval。

`Lost in the Middle` 是一个非常好的例子：模型拥有很长的 context window，并不代表它能够稳定地访问其中任意位置的信息；相关信息放在 context 中间时，性能可能明显下降。[MIT Direct](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638/119630/Lost-in-the-Middle-How-Language-Models-Use-Long?utm_source=chatgpt.com)

所以：

$\text{available} \neq \text{accessible} \neq \text{used}$



## Chain 2



$\boxed{ \text{Interface} \rightarrow \text{Architecture} \rightarrow \text{Representational Possibilities} \rightarrow \text{Learned Geometry} }$

其中：

### Tokenization / Interface

决定：

> 世界首先被切成什么粒度的输入对象。
> 
> 

例如 tokenization 本身就可能造成某些结构难以处理；已有工作直接构造 adversarial tokenization cases 来暴露这种限制。[arXiv](https://arxiv.org/abs/2405.17067?utm_source=chatgpt.com)

---

### Architecture / Inductive Bias

决定：

> 什么结构天然容易表达、传播和学习。
> 
> 

---

### Expressivity / Equivalence Classes

决定：

> 什么东西理论上能够区分，什么东西会被压进同一个 equivalence class。
> 
> 

---

### Representation Geometry

决定：

> 在这个允许的空间里，模型最终实际把信息组织成什么样。
> 
> 

所以：

$\boxed{ Architecture \rightarrow Inductive\ Bias \rightarrow Expressivity }$

和：

$\boxed{ Architecture \rightarrow Representation\ Geometry }$

是两条并行的 upstream 路径。

---



## Chain 3

$\boxed{ \text{Architecture} + \text{Data} + \text{Objective} + \text{Scale} \rightarrow \text{Training Dynamics} \rightarrow \text{Learned Internal Organization} }$

这里的关键是：

> Architecture 告诉模型“可以怎么学”，
> Data 告诉模型“经常遇到什么”，
> Objective 告诉模型“什么算成功”，
> Scale 决定“有多少 capacity / compute 可以用”，
> Training Dynamics 决定“最终实际上形成了什么”。
> 
> 

Scaling 本身不是直接制造某个 representation 的魔法；它通过改变 optimization landscape、capacity、data/compute allocation 等条件，使不同内部结构可能形成。Kaplan 和 Chinchilla 分别显示 model/data/compute 的 scaling 关系以及 compute\-optimal 的 model/data 配比。[arXiv](https://arxiv.org/abs/2001.08361?utm_source=chatgpt.com)

而 grokking 又提醒我们：

> “训练到某个 loss”并不等于“内部结构已经稳定形成”。
> 
> 

模型可以在已经过拟合之后继续发生结构性的 generalization change。[arXiv](https://arxiv.org/abs/2201.02177?utm_source=chatgpt.com)

所以 Training Dynamics 是一个非常重要的**桥梁节点**：

$\boxed{ \text{Training Conditions} \rightarrow \text{Internal Structure Formation} }$

---

## Chain 4

$\text{Representation} \rightarrow \text{Stored Knowledge} \rightarrow \text{Structured World Model}$

但这里要非常谨慎。

### Parametric Knowledge

问的是：

> 模型参数里是否编码了某些 factual / relational information？
> 
> 

LAMA 一类工作已经证明 pretrained models 可以包含相当多的 relational knowledge；ROME 则进一步尝试定位和编辑 factual associations。[ACL Anthology](https://aclanthology.org/D19-1250/?utm_source=chatgpt.com)

---

### World / Latent Semantics

则是更高一层：

> 这些 facts 有没有被组织成一个能够支持实体、关系、事件、变化、因果等结构性推理的 latent model？
> 
> 

所以：

$\boxed{ \text{having facts} \neq \text{having a world model} }$

这一区别非常重要。

否则很容易：

> probe 出一堆 factual information
> 
> 

↓

> 宣称模型形成了 world model。
> 
> 

这中间其实缺了非常大的一步。

---

## Chain 5

Circuits 又是连接 Representation、Knowledge、World Model、Capability 的桥

这里有一条我觉得很漂亮的关系：

$\boxed{ Representation \rightarrow Circuit \rightarrow Computation \rightarrow Capability }$

而知识则是：

$\boxed{ Knowledge \rightarrow Circuit\ Access \rightarrow Task\ Computation }$

所以一个模型：

> “知道巴黎是法国首都”
> 
> 

和：

> “在当前问题中正确调用这个知识”
> 
> 

本身是两个问题。

ROME 的工作已经体现了“knowledge storage”和“recall”之间需要区分：它研究的是 factual associations 的 storage / recall，并通过 causal intervention 来定位相关 computation。[arXiv](https://arxiv.org/abs/2202.05262?utm_source=chatgpt.com)



---

## Chain 6



ICL / Test\-Time Compute 则应该放在这张图的“运行时层”

这两个不是单纯的 capability。

它们是在问：

> **固定参数之外，模型还能不能在当前输入上动态形成新的计算过程？**
> 
> 

例如 ICL：

$\text{Fixed Parameters} + \text{Context} \rightarrow \text{Temporary Adaptation}$

已有 induction\-head 工作把 ICL 与具体的 match\-and\-copy circuit 联系起来；另一些工作则在受控 regression setting 下展示 Transformer forward pass 可以实现类似 gradient descent 的过程。[Proceedings of Machine Learning Research](https://proceedings.mlr.press/v235/singh24c.html?utm_source=chatgpt.com)

所以 ICL 实际上同时依赖：

$\text{Representation} + \text{Context Access} + \text{Circuit} + \text{Temporary State}$

这再次说明为什么把这些节点分开之后，真正的上下游关系会变得很明显。

---

Test\-Time Computation 则是在更高层：

$\text{Runtime Budget} \rightarrow \text{More Computation} \rightarrow \text{Potentially Better Capability}$

而且增加 test\-time compute 的收益依赖于 problem difficulty，不是简单地“多算就一定更好”。[arXiv](https://arxiv.org/abs/2408.03314?utm_source=chatgpt.com)

因此：

> **ICL 更像“运行时学习/适应”，Test\-Time Compute 更像“运行时增加 computation”。**
> 
> 

两者都属于 runtime substrate，而不是和 Representation、Memory 同层的内部对象。

---

## Chain 7

Conditional Computation 是一个特殊节点：它横跨“substrate”和“runtime”

MoE / Switch Transformer 是很好的例子。

普通 dense model：

$x \rightarrow \text{same parameters}$

Conditional computation：

$x \rightarrow \text{routing} \rightarrow E_i$

不同输入可以调用不同 parameters / experts。Switch Transformer 就明确把 routing 作为 sparse computation 的核心机制。[arXiv](https://arxiv.org/abs/2101.03961?utm_source=chatgpt.com)

所以它实际上连接：

$\text{Architecture} \leftrightarrow \text{Routing} \leftrightarrow \text{Representation} \leftrightarrow \text{Circuit} \leftrightarrow \text{Capability}$

甚至可能影响：

> “不同输入究竟是否共享同一个内部 state / representation space？”
> 
> 

因此它不能简单放在“模型如何计算”里面就结束了。





## 中间的重要环路

闭环一：Architecture → Representation → Capability → Architecture

$\boxed{ Architecture \rightarrow Inductive\ Bias \rightarrow Representation \rightarrow Capability \rightarrow \text{暴露 substrate limitation} \rightarrow Architecture' }$

这正好对应你前面提到的那类：

> “这个 representation 在这个 architecture 上天然更有希望。”
> 
> 

所以为什么 diffusion 的 harmonic representation 那类文章对 wcf 有价值？

不是因为它告诉你一个具体 representation。

而是因为它提供：

> **Architecture / substrate → representation geometry → capability**
> 
> 

这一整条 high\-level intuition。

---

闭环二：Representation → Memory → Access → Computation → Representation

$\boxed{ Representation \rightarrow Memory \rightarrow Access \rightarrow Computation \rightarrow New\ Representation }$



---

闭环三：Data → Training → Representation → Capability → Data / Objective

$\boxed{ Data \rightarrow Training \rightarrow Representation \rightarrow Capability \rightarrow 重新定义需要什么 Data / Objective }$

这就是为什么 Data、Objective、Training Dynamics 不能只被理解成“训练细节”。

它们可能决定：

> **模型究竟会自然形成什么内部世界。**
> 
> 







