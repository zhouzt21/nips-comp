# CNP 与 AttentiveCNP：原文、作者实现和 PAIEC 移植映射

记录日期：2026-10-22。范围：用户 `doc/复现与组合.md` 要求的 CNP 核心及 target-to-context attention。本文是来源和实现合同，不报告尚未运行的比赛增益。原始搜索、论文、作者代码及许可放在 `D:/nips-comp/paiec-dataset/reproduction_sources/cnp_anp/`；文件校验见同目录及本目录的 `cnp_anp_sources.json`。

## 一手来源

1. Garnelo et al., **Conditional Neural Processes**, *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80:1704–1713, 2018。正式版本：[proceedings](https://proceedings.mlr.press/v80/garnelo18a.html)、[PDF](https://proceedings.mlr.press/v80/garnelo18a/garnelo18a.pdf)。本地 `cnp_paper.pdf`、`cnp_paper.txt`。方法定位：§2.2、公式 (1)–(3)，训练定位 §2.3、公式 (4)，均在 PDF 第 3 页（论文页 1706）。分类输出 logits 的许可也明确写在 §2.2。
2. Kim et al., **Attentive Neural Processes**, *International Conference on Learning Representations*, 2019。作者原文 [arXiv:1901.05761v2](https://arxiv.org/abs/1901.05761v2)，版本日期 2019-07-09；PDF 页眉注明 ICLR 2019 conference paper。本地 `anp_paper.pdf`、`anp_paper.txt`。方法定位：§2.1、公式 (1)–(3)（PDF 页 2–3），§2.2 scaled dot-product attention（页 3），§3/Fig.2 deterministic cross-attention 和 latent path（页 4），§4 的 1D 实验（页 5）。本次读取的是该作者公开全文，不把 arXiv 版本本身称作 proceedings 文件。
3. 作者仓库 [google-deepmind/neural-processes](https://github.com/google-deepmind/neural-processes)，固定 commit [`aca7e38ea64b718fbd7f311ccae5d51d73447d15`](https://github.com/google-deepmind/neural-processes/tree/aca7e38ea64b718fbd7f311ccae5d51d73447d15)，提交时间 2020-03-20T12:57:17Z。原始 notebooks、README、Apache-2.0 LICENSE 均保留在本地 `neural-processes/`。从 notebook code cells 顺序导出的 `.py` 仅供逐行阅读，没有替换作者源文件。版权为 2018/2019 Google LLC。

检索全部使用 AnySearch：先 `get_sub_domains --domains academic,code`，再 `academic.search` 标题查询；用 `extract` 读取正式 proceedings 页面和作者仓库。PDF 的 AnySearch `extract` 两次返回 `extract_failed`，原始失败记录已保留；随后直接下载已经由这些一手入口定位的 PDF，以本机 `pdftotext -layout` 提取并读取全文方法部分。这是确定 URL 的源文件下载，不是更换搜索引擎。作者仓库使用 `git clone --depth 1` 下载并记录 commit。

## 实现对应关系

本项目实现：`paiec_baseline/mlp/neural_process.py`。以下 notebook cell index 均从 0 开始；导出 `.py` 的行号可用于本地核对。

| 原文机制 | 作者实现定位（固定 commit） | 本地移植 |
|---|---|---|
| 每个 context pair 的表示 `r_i=h(x_i,y_i)`，CNP (1) | `conditional_neural_process.ipynb` cell 9，`DeterministicEncoder.__call__`；导出 `.py:187` | `CNP.context_encoder`，拼接 `x` 和对应 `y`；四层 dense，前三层 ReLU，末层线性 |
| 集合 mean，CNP (2) | 同 cell 9，导出 `.py:225` 的 `tf.reduce_mean` | `CNP._aggregate`，只在 context 维度求均值并复制到 query |
| target decoder，CNP (3) | CNP notebook cell 11，`DeterministicDecoder.__call__`；导出 `.py:241` | 依作者顺序拼接 `[r,target_x]`，两层 ReLU 后输出一个 Bernoulli logit |
| 随机 context 条件 likelihood，CNP (4) | CNP notebook cell 21，导出 `.py:408`，`-tf.reduce_mean(log_prob)` | 模块暴露 `predict_logits`；公共 harness 用 Bernoulli BCE/NLL，最终仍按 PAIEC Brier/ALC 评价 |
| pointwise MLP 最后一层不加 ReLU | `attentive_neural_process.ipynb` cell 6，`batch_mlp`；导出 `.py:189` | `_mlp` 的线性层/激活顺序一致 |
| deterministic value encoder | ANP notebook cell 8，`DeterministicEncoder.__call__`；导出 `.py:234` | 与 CNP 相同的 `[x,y]` pointwise encoder，输出 values |
| `softmax(QK^T/sqrt(d_k))V`，ANP §2.2 | ANP notebook cell 16，`dot_product_attention`；导出 `.py:501` | `dot_product_attention`，softmax 在 context 轴，未对 query 做归一化 |
| keys 与 queries 共享输入 MLP | 同 cell 16，`Attention(rep='mlp',att_type='dot_product',normalise=True)`；导出 `.py:601`–`615` | `AttentiveCNP.key_query_encoder` 同一实例处理 context_x 与 target_x |
| 空 context 的固定表示 | ANP §2.1，PDF 页 2，`r_empty` 为固定向量 | `N=0` 返回全零表示，不对空集合做 mean/softmax；decoder 可形成 item prior |

作者 notebook 的默认 attention 示例为 multihead；但同一个官方实现明确提供 normalized dot-product 分支，本次选择该分支作为最小 cross-attention 机制。没有在来源映射中将 dot-product 说成作者默认最强设置。

## 固定架构与公平比较边界

共同构造：`input_dim=640, hidden_dim=64, representation_dim=64`。输入为 `[MiniLM384,metadata256]`，归一化参数由公共 harness 的 fit 数据决定。模型本身没有训练数据读取、benchmark 划分或标签查询逻辑。

- Encoder：`[641,64,64,64,64]`；Decoder：`[704,64,64,1]`。
- CNP：mean aggregation，**102,913** 个参数。
- AttentiveCNP：另有共享 key/query MLP `[640,64,64]`，**148,097** 个参数，增加 **45,184**。两者保持相同基础编码器、解码器和层宽；不是等参数比较。因此差异不能直接归因为 attention 权重的单一因果效果。
- `forward(context_x[B,N,D],context_y[B,N],target_x[B,Q,D]) -> probabilities[B,Q]`；`predict_logits` 返回同形状 logits。批次中 context 数相同，无 padding mask；不同预算由 harness 分批调用。模型不会读取 target 标签。
- 最终逐响应 Brier/ALC 的权重、重复 response 的保留、pair 等权、六预算和数据拆分由公共 harness 保证，不能由本模块单独证明。

## 与完整原论文复现的差异

**这是 CNP 的二元任务移植和 deterministic cross-attentive CNP 核心复现，名称固定为 `CNP`、`AttentiveCNP`。完整 latent ANP 没有在此实现。**

1. 作者 notebook 的 GP 回归输出 Gaussian mean/std；此任务输出 Bernoulli logits。CNP 原文 §2.2 支持 categorical logits，二元形式与该机制一致；本次没有复现论文 GP/图像定量结果。
2. 原论文/作者代码的 latent ANP 包含 global latent、context/target 后验和 ELBO/KL。当前模型没有 latent 或 KL，不宣称具有 ANP 的函数级随机性或联合不确定性。
3. 本次没有 self-attention；ANP §4 的 1D GP 实验本来就没有使用 self-attention。因此命名为 AttentiveCNP 的关键是缺少 latent path，不是声称所有 ANP 都必须有 self-attention。
4. 原 CNP 示例层宽 128，本次固定 64。代码结构保持 pointwise dense/ReLU，而参数规模和训练数据按 PAIEC 调整。
5. CNP 原式 (4) 的 target 训练集合含已观察和未观察项，ANP 通常 `C` 是 `T` 的子集。公共实验若只在与 context 不相交的 query 上计算 NLL，则是对竞赛 held-out query 目标的训练协议适配，需在运行合同中记录。
6. 完整 latent ANP、自注意力和多头模块是后续有进入条件的工作；本次核心试验失败不能直接否定这些未实现部分。

## 已执行验证和证据边界

用迁移后的 Python 3.12 / Torch 环境运行 `tests/test_neural_process.py`，**7 项检查全部通过，运行耗时 12.039 秒**。命令及状态记录于 `cnp_anp_sources.json`：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:/nips-comp/paiec-runtime-win64-py312-cpu/python.ps1 -c "import sys, unittest; sys.path.insert(0, 'paiec_baseline'); suite=unittest.defaultTestLoader.discover('tests', pattern='test_neural_process.py'); result=unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(not result.wasSuccessful())"
```

验证覆盖独立 NumPy dense/mean/scaled-attention/sigmoid 方程对照（float64，`atol=rtol=1e-12`）、context 同步置换不变性、target 重排等变性、逐 query 与批量 query 相同、episode 批次独立、B0/1/3/7/15/31 数值有效、支持标签翻转/配对置换能够改变输出、标签梯度非零、BCE 更新和导出重载一致，以及错误 shape 拒绝。

这些检查证明实现结构和数值契约成立；**随机初始化模型对标签变化有响应不等于真实训练模型已经学会利用对应关系**。后者需由公共真实实验的 support-label permutation 对照判断。

作者 README 指明其运行依赖 TensorFlow 1.13.1；迁移环境没有 TensorFlow。本次没有安装旧 TF 或运行完整原 notebook，所以证据等级是“作者源静态逐块对齐 + 独立 NumPy 方程数值一致”，不是“原 TF checkpoint 跨框架逐值对齐”。
