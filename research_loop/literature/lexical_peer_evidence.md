# R69–R72 原始证据与实现对应（2026-10-23）

所有新增发现检索使用 AnySearch；搜索摘要仅用于定位，以下方法依据均已读取正文/源码。原始抓取失败和修复路径也保留，未用摘要填补缺口。

|依据|版本与精确定位|本轮使用|限制|
|---|---|---|---|
|Busing, *Monotone Regression: A Simple and Fast O(n) PAVA Implementation*, JSS 102 Code Snippet1, 2022，正式期刊code snippet|论文p1 Eq(1)，p4–5 Algorithm1；作者Leiden存档Publisher Version，PDF SHA256 `15328aa2407c21cc409222d0eb01285338b6497d1268fa53fab15f7a567ce478`|加权平方目标与PAVA，R70将Brier二次项完成平方后使用非增版本|原文不主张PAIEC预算系数应非增；这是本轮可证伪假设|
|作者CRAN monotone 0.1.2|官方CRAN镜像repo tree `36c17fa4b03ac20b8d0a801c9f75ba83b68fe65c`，`src/monotoneC.c`，作者说明与算法weighted merge已逐项阅读|核对算法1的向上/向下合并与块恢复|本轮直接调用SciPy，不复制作者C代码或重新跑其速度benchmark|
|SciPy v1.13.1官方源码|`scipy/optimize/_isotonic.py`，isotonic_regression；Notes与reference[1]明确Busing2022|运行实际安装的 `isotonic_regression(target,weights=ALC_weight*C,increasing=False)`，最后clip[0,1]|仅重用该子算法，非整篇论文结果复现|
|Gulrajani & Lopez-Paz, *In Search of Lost Domain Generalization*, ICLR2021|原文arxiv2007.01434，§3–3.1；已存 `D:/nips-comp/paiec-dataset/training_mechanism_sources/domainbed.txt`；作者DomainBed固定commit `b93c22a1cfc3b2428398272c1a116c8de1f4139e` 的 model_selection.py|来源Benchmark分组模型选择；说明selection是方法的一部分|本轮不是DomainBed图像数据集复现，不能据此断言alpha可迁移|
|PAIEC官方接口与baseline repo|官网2026-10-23提取；repo `cc88a959c1e5e7462f31a9215baba066ce6d6e2b`；`empirical_mean/model.py`、`tools/streaming_ingestion.py`|真实可见字段、预算、完整subject匹配、共享证据入口、事件式evaluation|客户端不含私有coordinator采样规则；无真实hosted运行证据|
|本项目R25/R64、R20/R36|冻结 `run_metadata_round.prediction`、`context_tokens.template/retained`、`conditional_components`、`pooled_item_slope.prepare`|完整继承中心化ridge、当前support模板和within-subject pooled direction|是本项目既有代码；新的预算/分组收缩及peer分解为自有实验算子，不假称外部论文原法|

原始入口：

- 论文：https://doi.org/10.18637/jss.v102.c01
- 作者原文：https://scholarlypublications.universiteitleiden.nl/download/1887/3515640/1?col=81522199280002711
- 作者代码发行：https://cran.r-project.org/package=monotone （源码固定镜像 https://github.com/cran/monotone/tree/36c17fa4b03ac20b8d0a801c9f75ba83b68fe65c ）
- SciPy：https://github.com/scipy/scipy/blob/v1.13.1/scipy/optimize/_isotonic.py
- DomainBed：https://arxiv.org/abs/2007.01434 ，https://github.com/facebookresearch/DomainBed
- PAIEC：https://aims.stanford.edu/competition/ ，https://github.com/aims-foundations/paiec_baseline

原文证据保存在 `paiec_baseline/runs/research_loop/R69/`；retrieval JSON含URL、内容hash与失败记录。直接JSS访问TLS失败后，经AnySearch定位作者大学原文下载成功；前两次本地PDF解析依赖/PATH错误已改用现有TeXLive pdftotext，没有安装降级依赖。
