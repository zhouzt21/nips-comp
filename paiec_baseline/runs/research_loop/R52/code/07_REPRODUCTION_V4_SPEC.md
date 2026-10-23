# R51 起：基于原文与官方代码的复现及 PAIEC 迁移

开始：2026-10-22。用户执行授权与完整方案原文保存在 `doc/复现与组合.md`；该文件保持原文，不用本记录替代。起点代码提交：`fd5b06df51233392ea9d482732af8063f4222ec1`。

## 问题、主张与证据合同

任务类型：方法核心复现、受控迁移实验和失效分析。问题 A 是集合条件模型能否更有效利用当前预算内支持标签；问题 B 是内容难度与 subject 条件能力能否提供跨 benchmark 可迁移的零/低标签先验。当前不预先主张任何方法有正收益。

可证伪假设：CNP/ANP 的训练目标与支持集合机制能够在相同信息下改善 R25；轻量 amortized IRT 的因子分解能够改善 held-out benchmark 的 B0/B1。候选必须接受相同预算的强简单基线、R25、支持标签对应关系破坏控制及 benchmark 分组验证。用较少层数和固定计算预算执行的 PAIEC 迁移，不等同于复现原论文的全部数据集及数值。

检索使用 AnySearch；论文方法、假设、公式和作者代码须逐项核对。每条路线保存原文来源、正式版本/预印本状态、仓库 commit、文件/函数定位、原始实现与 PAIEC 适配差异。检索摘要只用于寻找来源。研究结束保存真实分数、误差/机制分析和负结果，不能以实现存在替代实验完成。

## 冻结约束与进入条件

- 数据 revision：`54074453da7b9911be0f48e4c560be547bc84d21`；MiniLM revision：`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。复用 R25 的原始 outer benchmark 划分和合法输入；不以新随机种子或复用公开 benchmark 声称独立盲测。
- 预算 `0,1,3,7,15,31`，ALC 权重 `.1,.2,.2,.2,.2,.1`，pair 等权；distinct item 隔离支持与评价，保留同一 item 全部评价 responses。来源模型选择只能访问 outer training benchmarks；特征变换也在训练侧拟合。
- 每条路线先做来源内 benchmark 分组训练/验证，再按预先登记条件决定是否进入全部 5,242 pairs / 三个原种子评价。每个模型的架构、优化、seed、训练暴露量、最大计算预算和 stopping rule 在评分前写入对应协议；不根据 outer 结果扩大搜索。
- 原 Research Gate 保持：相对冻结 smoothed mean strength 2 和训练内选择的强简单基线，pair-equal ALC 改善至少 0.005，三折均改善，benchmark-equal 改善，benchmark 聚类 paired bootstrap 95% 上界小于零，解释超过 0.03 的 benchmark 退化。
- CNP/ANP 与 IRT 先独立判断；只有各自有证据支持的增益才进入组合。若表示不足，进入 ADeLe/General Scales 原文与代码核对及可部署特征可行性分析；其后实验同样先冻结合同。IrtNet MoE、完整 DeBERTa assessor 和共享标签模型保留原方案中的条件进入要求。
- CPU 环境：AMD Ryzen 7 4800H，8 cores / 16 threads，约 16 GiB RAM；开始时可用内存约 1.3 GiB。研究与实现并行，训练进程串行，限制批大小和线程数；不修改全局环境或占用外部付费推理 API。

## 状态与交付

当前：核对原文/代码、构建共同 episode 和评价接口。轮次 R51 用于 CNP/ANP，R52 用于轻量 IRT；后续轮次由实际进入条件决定。

交付：逐条来源映射、冻结协议、模型/训练/评价代码、原始 JSON 和训练日志、来源内与外层证据范围、结果及理解汇总、当前运行环境下的复现命令。未来汇报状态同步 `D:/todo/tasks/graph/nips-comp.paper-reproduction-2026-10-22.md`。未完成事项必须保留明确状态，不能把门槛失败记为性能目标达成。
