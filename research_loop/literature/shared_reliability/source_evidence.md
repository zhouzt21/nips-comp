# plan_9 原文与作者代码依据

2026-10-23。窄范围AnySearch检索/提取保存在本目录JSON；检索摘要只作发现。MTL标题查询返回相关工作而未精确命中KDD2004，因此其证据继续采用已下载核验的作者原文，不以搜索结果替代。

|依据|精确位置/版本|本轮使用与限制|
|---|---|---|
|Evgeniou & Pontil, Regularized Multi-Task Learning，KDD2004，同行评审|作者PDF http://www0.cs.ucl.ac.uk/staff/m.pontil/reading/mt-kdd.pdf ，R75/literature/Evgeniou_Pontil_KDD2004.txt，Section2 Eq1–3；原PDF/文本hash见R75 manifest|共同参数+任务偏移收缩的结构依据；原论文hinge loss不等于PAIEC Brier。本轮直接复用已验证R76平方损失算子，不改变罚项8。|
|MALSAR作者实现|jiayuzhou/MALSAR commit fb9751594983df020ddc4f7e4a40520ee7c37989，MALSAR/functions/SRMTL/Least_SRMTL.m，头部OBJECTIVE及RRt/梯度；manual4.4|0.5*SSE、R=I−11'/T结构罚项及L2转换沿用SHARED_SLOPE_SOURCES。复查作者源码，不运行外部MEX。|
|Gulrajani & Lopez-Paz, In Search of Lost Domain Generalization，ICLR2021，同行评审|arxiv2007.01434，正文§3、§3.1；D:/nips-comp/paiec-dataset/training_mechanism_sources/domainbed.pdf与domainbed.txt第160行起|模型选择属于学习方法，区分training-domain validation、leave-one-domain-out、oracle。支持明确来源选择合同，不保证support风险可迁移。|
|DomainBed作者代码|facebookresearch/DomainBed commit b93c22a1cfc3b2428398272c1a116c8de1f4139e（本机git HEAD核对），domainbed/model_selection.py 的IIDAccuracySelectionMethod和LeaveOneOutSelectionMethod|核查validation与test分开、source环境聚合逻辑；本轮用自有Benchmark分组协议与Brier，不声称运行DomainBed图像基准或完全复制其leave-one-domain-out。|
|本项目冻结算子|shared_slope_model.py的predict/centered_arrays/cohort_map；R75/R76 contract中的源码hash|R77复用实际中心化核模型、clip、own mean；R78的支持mask和可靠性变量是预设机制实验，文献没有预言它们必然有效。|

R78/R79严格source选择细节见SHARED_RELIABILITY_SPEC及OOF_ADDENDUM。不同cohort比较的bootstrap保持同BM重复测量依赖；这不会恢复公开数据被多轮选择后的盲测属性。未新做广泛综述或访问Hosted。
