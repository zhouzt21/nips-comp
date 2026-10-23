# plan_2 实现与复现合同核对

2026-10-22。主agent执行数值对照，另使用一个只读structural reviewer审计复杂复现合同；没有双审或Quality Judge验收。

## 数值证据

入口：`paiec_baseline/mlp/run/audit_training_mechanism.py`。

- 三层logit初始化处Taylor展开 vs Torch autograd JVP：RMSE差 `7.892991815694472e-17`。
- pair/Benchmark风险权重与手算两级平均一致；两权重行均值为1。
- `abstract_reason` seed73重建的R43 `z/base/delta/fraction/weight` 与历史缓存逐元素完全一致。
- seed5501与固定seed73保持相同pair和query行数，支持/查询内容发生变化，身份不交断言通过。
- epoch0 R43 vs R25六预算曲线最大误差 `6.589387369082544e-09`。
- MLP按interactor变体聚合 vs历史逐响应评分最大误差 `2.3158305939574575e-08`。
- 每个真实fit逐epoch要求Torch到NumPy预测误差<1e-5；完整记录见history/export_error。实际fit资源及checkpoint数量见 `completion_audit.json`。
- R54前20epochs与历史R43合并来源曲线另行核对：query组最大差 `1.56e-6`，episode组约 `2e-9`；六个最终选择epoch均一致（query0/0/20，episode0/0/9）。因此不宣称跨机器历史训练逐bit回放；本轮固定/重采样使用相同新运行的初始化、标准化和步数控制，进入阈值为1e-5。完整差值见 `result_audit.json`。

## 只读审查及修正

审查确认：来源Benchmark互斥、fit-only标准化、历史优化器与损失一致、重复响应权重、pair加权共同epoch选择、Taylor/CKA方程、R55匹配步数与预算内置换差分、R56仅改风险权重、R57参数/概率平均、R58条件逻辑。

发现并收口的三项：

1. **R43历史名称**：历史固定20epochs，无patience；新增patience为反事实对照。`TRAINING_MECHANISM_V5_ERRATA.md`明确记录。
2. **资源/异常状态**：核心训练器在epoch开头检查1800秒，新增stage结束逐fit审计拒绝最终epoch越限，并在异常时落incomplete/attempt。R54实际每fit均远低1800秒，无超限结果计入负证据。
3. **恢复来源约束**：新增全部历史依赖与split快照、输入NPZ内容清单；阶段源码在首次执行冻结，复用前和结束后均校验。最终单点复核guard SHA256 `4f2d383bc0b1ad54b6f2d29b9fa637d3ecd6157700cbe58ad276f994f9c6b208`，审查回复无剩余阻断。

时间边界：R54每fit原有核心/spec hash先于训练；更广依赖快照在R54结束后补齐，不能倒写成训练前已冻结所有依赖。期间核心训练/data源没有改变，审查当时记录的hash与快照一致。后续R55起使用完整guard。

## 运行异常

R55首次准备因随机新题缺少迁移MiniLM缓存失败，**尚无fit**。失败保留于 `execution_attempts/resample_missing_cache.*`。补算用原 `SemanticFeatures` 和已迁移模型，`local_files_only=True`；已存在key直接读取，新增key单独记录。Tokenizer可能对截断前原始长序列给出警告；原编码器随后固定保留96头token+30尾token，加特殊token后最多128，实际模型输入未越界。未修改这一历史预处理。

这些审计保证实现与合同的一致性，不能将来源内选中的结果解释成无偏盲测或理论因果证明。

## 最终结果一致性

`mlp.run.audit_training_mechanism_results`已对真实完成产物核对：R55十二fit和R56六fit的epoch0权重、fit参考数据hash、固定验证pair身份与逐epoch更新数均匹配R54；R55每fit20个活动数据hash互异；R56活动训练行hash与pair风险逐epoch相同。所有36fit/996checkpoint完成资源审计，最长fit23.5秒。

R55、R56均0/3来源池通过全部条件；R57 query/episode的EMA、SWA、概率平均均未通过三个池共同条件，且均差于最佳单一checkpoint；MLP稳定窗口与R58初始化触发均不成立。报告包含99个方法/选择规则×来源池记录、594条六预算Brier/正确率记录及逐Benchmark表。没有新外层评价或提交；负结果按本轮配置解释。
