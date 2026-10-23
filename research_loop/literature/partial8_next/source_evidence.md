# plan_10 原始证据与实现映射

2026-10-23。检索/页面提取使用AnySearch，原始JSON在本目录。关键论据来自以下实际读取的全文/源码，不来自搜索摘要。

|来源|版本与精确定位|本轮用途/边界|
|---|---|---|
|Bengio & Grandvalet, No Unbiased Estimator of the Variance of K-Fold Cross-Validation|JMLR 5:1089–1105, 2004，同行评审；作者正式PDF `https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf`；§2.3 p1093、§5 Theorem6、§8.3 p1103；本地grandvalet04a.pdf/txt|CV估计的训练样本量与full fit不同，重叠fit产生相关误差，LOO仍无普适无偏方差估计。本轮不把固定OOF预测的item bootstrap称严格95%置信保证。论文没有预言PAIEC LOIO必然有效。|
|scikit-learn官方分组CV实现|作者仓库tag1.5.2，`sklearn/model_selection/_split.py`，GroupKFold类537行、分组分配605–636；LeaveOneGroupOut类1274行、分组留出1330–1337；官方说明cross_validation.html|按group整组留出，避免重复item跨fit/hold。R80 LOIO遵循同一group隔离逻辑；5fold采用预定随机group轮转，和该版本按group大小贪心分配并非逐行相同算法，不声称完全复制官方分组序列。|
|Evgeniou & Pontil Regularized Multi-Task Learning|KDD2004同行评审；原作者PDF/文本沿用R75归档，§2 Eq1–3|共享参数+任务偏移正则的结构依据。本项目平方损失/核/截距规则与原文hinge任务不同。|
|MALSAR作者Least_SRMTL.m|commit fb9751594983df020ddc4f7e4a40520ee7c37989；实际下载源码头部OBJECTIVE、R=I−11'/T、rho_L2与RRt|复核相关任务的结构正则，不运行MATLAB/MEX原基准。与R75/R76原始结构映射相容。|
|DomainBed|作者commit b93c22a1cfc3b2428398272c1a116c8de1f4139e；原文§3–3.1/model_selection.py已在plan9读取并保留hash|来源模型选择隔离纪律沿用；本轮无新DomainBed图像实验。|

本轮核心模型调用冻结shared_slope_model.py；raw包装只为分离shared/offset与clip审计，其clip输出必须与冻结预测数值一致。匹配位移成本与合成机制实验是本项目实验设计，不冒称原论文已经验证。

AnySearch的GitHub页面提取未包含完整_split.py源码，因此从该页面给出的官方raw入口下载固定tag代码，实际读取函数体。PDF先校验%PDF字节；SHA256/大小在original_manifest.json。首次提取发现迁移环境没有pypdf/fitz，使用本机既有`D:/texlive/2025/bin/windows/pdftotext.exe -layout`完成，不改运行环境依赖。
