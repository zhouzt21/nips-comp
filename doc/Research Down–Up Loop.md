# Research Down–Up Loop

## Research Down–Up Loop

```Markdown
┌──────────────────────────────┐
                 │  HIGH-LEVEL                  │
                 │  当前问题 / Claim / Bottleneck│
                 │  对“到底发生了什么”的判断      │
                 └──────────────┬───────────────┘
                                │
                           ① 下钻
                                ↓
                 ┌──────────────────────────────┐
                 │  MID-LEVEL                   │
                 │  mechanism / representation  │
                 │  information / architecture  │
                 │  task structure              │
                 └──────────────┬───────────────┘
                                │
                           ② 再下钻
                                ↓
                 ┌──────────────────────────────┐
                 │  LOW-LEVEL                   │
                 │  可观测现象 / intervention   │
                 │  setting / experiment        │
                 └──────────────┬───────────────┘
                                │
                         ③ 得到局部结果
                                ↓
                 support / anomaly / failure
                 / boundary / unexpected rescue
                                │
                                ↓
                         ④ 开始上溯
                                │
              ┌─────────────────┼─────────────────┐
              ↓                 ↓                 ↓
            思考              调研               交流
       alternative        field evidence     expert judgement
       explanation        model nature       reframing
              └─────────────────┼─────────────────┘
                                ↓
                 ┌──────────────────────────────┐
                 │  更新 HIGH-LEVEL             │
                 │  “原来问题可能应该这样表述”    │
                 │  “真正 bottleneck 可能在这里”  │
                 └──────────────┬───────────────┘
                                │
                         ⑤ 选择研究路线
                                ↓
          ┌────────────────┬───────────────┬────────────────┐
          │ Audit / Pivot  │ Encode by     │ Decompose +    │
          │ / Limit        │ Construction   │ differentiated │
          └────────────────┴───────────────┴────────────────┘
                                │
                                ↓
                       ⑥ 设计下一实验
                                │
                                └────────→ 再下钻
```

$\boxed{ \text{理解问题} \rightarrow \text{下钻} \rightarrow \text{观察} \rightarrow \text{上溯} \rightarrow \text{重新理解问题} \rightarrow \text{决定研究动作} \rightarrow \text{再次下钻} }$



### HIGH\-LEVEL STATE

先明确当前：

- 我认为真正的问题是什么？

- 当前 claim 是什么？

- bottleneck 是什么？

- 有哪些 competing explanations？

- 当前最可能的 contribution route 是什么？

↓

### DOWNWARD：把 high\-level 变成可观察问题

把：

> “我认为 X 是问题”
> 
> 

继续拆成：

> 如果 X 真成立，局部应该看到什么？
> 
> 

得到：

$H_{high} \rightarrow H_{mid} \rightarrow P_{observable}$

↓

### MICRO EXPERIMENT

设计最小但具有**高区分力**的实验。

关键不是：

> 实验够不够小。
> 
> 

而是：

> **实验结果出来后，能不能显著改变我对上层 hypothesis / search space 的判断？**
> 
> 

实验不是“验证答案”，而是“帮助你往上爬”

以前容易把实验理解成：

> 我已经有一个 hypothesis → 做实验 → 看它对不对。
> 
> 

但在探索阶段，更准确的是：

> **我先下钻到一个具体问题 → 做一个足够有区分力的实验 → 从结果中获得能让我重新理解上层问题的信息。**
> 
> 

“见微知著”它不是：实验越小越好。而是：

> **实验的 observation 可以很局部，但它产生的 inference 要尽可能能作用到更高层。**
> 
> 

比如：

```Plain Text
“这个 setting 失败了”
        ↓
“为什么失败？”
        ↓
“是这个 setting 的问题？”
        ↓
“是这一类 intervention 的问题？”
        ↓
“还是当前 representation 根本不适合表达这个东西？”
        ↓
“那么我们的 scientific object 是否应该重新表述？”
```

所以好的探索实验实际上是在做：

$\text{local observation} \rightarrow \text{higher-level hypothesis update}$

这就是你说的：

> **不仅要见微，还要尝试通过思考、调研和妥善设计后续实验向 high\-level 反溯。**
> 
> 

我认为这句话应该成为整个循环的中心。



↓

### LOCAL RESULT

得到：

- support；

- failure；

- anomaly；

- unexpected rescue；

- boundary；

- conflicting results。

**不要马上继续做同层 variant。**

↓

### UPWARD DIAGNOSIS

问：

> 这个结果究竟告诉了哪一层？
> 
> 

依次检查：

```Plain Text
implementation failure?
      ↓
experiment-design failure?
      ↓
method failure?
      ↓
mechanism failure?
      ↓
representation failure?
      ↓
information insufficiency?
      ↓
architecture/substrate mismatch?
      ↓
task/problem formulation failure?
```

↓

### TRIANGLE：思考 × 调研 × 交流

此时三个东西共同帮助你上溯：

#### **思考**

> 有没有 alternative explanation？
> 
> 

#### **调研**

##### “调研”在这个循环里其实有两种完全不同的作用

###### 第一种：Problem\-local literature

回答：

> **这个问题在领域里已经走到哪里了？**
> 
> 

它提供：

- 已知现象；

- 已知方法；

- 已知 failure；

- 已知 limitation；

- novelty boundary；

- 哪些解释已经被排除。

这主要服务于：What is the current field\-level problem?

---

###### 第二种：Model\-nature / Architecture\-nature literature

> 这个领域 / model family 有没有类似结构？
> 
> 

尤其调用长期积累的：

> architecture / representation / inductive\-bias / limit knowledge。
> 
> 

> **“这个模型/架构天然倾向于形成什么样的 representation？它容易做什么、不容易做什么？它的 inductive bias 是什么？”**
> 
> 

```Plain Text
这个 model family
      ↓
存在某种特殊 inductive bias
      ↓
会自然产生某类 representation
      ↓
某些结构在这个 substrate 中可能更“自然”
      ↓
所以如果我要研究某种 representation / state，
换一个 substrate 可能值得考虑
```



#### **交流**

> 一个经验更高的人会不会看到我们没看到的变量？
> 
> 

↓

### CLIMB BACK

不是：

> 找到一个更细的解释。
> 
> 

而是：

> **重新命名当前问题的 high\-level identity。**
> 
> 

即：

$\text{local result} \rightarrow \text{new problem representation}$

↓

### CHOOSE CONTRIBUTION ROUTE

```Plain Text
A. Audit / Pivot / Limit?
B. Encode Structure by Construction?
C. Decompose + Differentiated Treatment?
```

#### Research route 是可组合的 operator sequence



当前的 Research route 不应理解成一个项目只能选择一种 route，然后一直沿着这条路线做下去。

更准确的理解是：**Research route 是一组可以按当前 diagnosis 依次调用、组合和切换的 research operators。**

$\boxed{O_1 \rightarrow O_2 \rightarrow O_3 \rightarrow \cdots}$

其中每一个 operator 都对当前问题做一次不同类型的变换：



- 改变我们对问题的表述；

- 审计一个隐藏 assumption；

- 建立一个新的 measurement；

- 证明两个看似不同的东西其实等价；

- 把某种结构直接编码进 architecture；

- 制造新的 supervisory signal；

- 将不同子问题交给不同 solver。

    

因此：



> **route 更像“operator sequence”，而不是互斥的路线类别。**
> 
> 



一个 operator 的输出可以成为下一个 operator 的输入：

$\text{Diagnosis}
\overset{O_1}{\longrightarrow}
\text{New Object}
\overset{O_2}{\longrightarrow}
\text{New Constraint}
\overset{O_3}{\longrightarrow}
\text{Research Move}$



详见：[常见的组合思路](https://my.feishu.cn/wiki/UfOhwDYPEimQU1kcZOjcTht9nLf)



### NEW HIGH\-LEVEL CLAIM

形成新的：$H'_{high}$ 然后重新下钻。

于是：

$\boxed{ H_{high} \rightarrow Experiment \rightarrow Observation \rightarrow Diagnosis \rightarrow Literature/Discussion \rightarrow H'_{high} \rightarrow Route \rightarrow Experiment \rightarrow \cdots }$

