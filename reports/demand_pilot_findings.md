# R74 No-Peer：四维任务需求与 CPU 学生来源 pilot

日期：2026-10-23。状态：完成来源 pilot；按冻结条件停止，未运行新的 5,242-pair 外层。原始 R25 / R56 / R72 与 fallback 未修改。

## 结论

本轮**没有证明 Luna 四维需求标注在现有输入之外提供可迁移的 response 预测价值**。四维完整 query 子集的教师增强 B0 Brier 为 **0.337674293**，同数据 base 为 **0.249250842**，置换教师为 **0.310736254**。教师对 base 的 Δ=+0.088423451，95% BM-cluster CI=[0.03778891512511477, 0.1284794342287258]；对置换 Δ=+0.026938039。预注册教师进入条件未通过。

CPU ridge 学生完成训练和实际六预算评价。其 grade MAE **0.991676**，fit-mean **0.999972**；提升很小。学生低预算先验比同容量 base 的来源 ALC 改善 **0.000079265**（CI=[-0.00029523544442622865, 0.0001308795269741821]），比置换学生改善 **0.000137498**（CI=[-0.0002839372242164433, 1.8488609697119316e-05]）。这些是小幅来源信号，不能抵消教师信息检验失败，也不能冒充完整外层收益。

## 原文、作者代码与适配边界

[Zhou et al., Nature 2026](https://www.nature.com/articles/s41586-026-10303-2) 原文 Methods 的 LLM annotators、Assessors and metrics 和补充 Table 23，及[固定作者仓库](https://github.com/Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation/tree/816321f327dfe66910b72b514156aaa296b15f00)是依据。`src/adele/analysis/prediction.py` 保留 subject-specific RF（100 trees、min_samples_split=50、entropy）、Benchmark grouped OOF、complete-case需求处理。原论文使用18维+UG和GPT-4o，而本轮是原4rubrics + Luna、无UG、跨subject共享的线性ridge先验及4个蒸馏head。**这是有原文依据的机制适配，不是原论文准确率的逐参数复现。** 不使用作者 battery 的 response 数据。

## 数据与合法性

- 原 R53 的128 distinct query items；436/512维度有效、108个四维完整。grade评价872条是各有效item-rubric在两个重叠pool中的暴露，只有110个至少一维有效的unique items；并不是872独立标注。源标注由root在看response结果前定稿；null不当成0。教师用完整case，学生按维度忽略null。同模型Luna的重复QC是重复性信息，不是人工校准的正确性。
- 三个原来源pool分别有316/260/248个pair，合计824 exposure、412 distinct pairs；25上限/BM、seed73、<=64query。三个pool重叠，不能称独立三次重复。教师子集有176 exposure、88 distinct pairs，24个BM；其分数不能和完整来源或5,242外层直接比较。
- 每个pool内原 `inner_halves` 双向BM OOF；student和response ridge仅fit半组。outer BM排除，fit IDs/BM清单在 `isolation_audit.json` 和各inner目录。响应训练pair-equal、pair内保留重复response权重。
- B0教师只评价已注释query。大多数support没有teachergrade，故**没有纯teacher B1或teacher局部六预算结果**。实际学生可给任何已有640表示的item计算四维，完整六预算单独评价。
- R56是原R59的OOF；R66 item是R67的来源原型，不是R67 outer-refit预测。两者训练样本/模型不同，只作强历史参照；需求增量由同数据644槽base/teacher/permutation和student比较支持。

## 实际分数与正确率

完整来源 R25 ALC=0.211347421；prior_student=0.210595327；R56_segment=0.209950904；R66_item_segment=0.209609263。学生先验仍弱于历史两种先验。其 B0/B1准确率为 54.442% / 63.281%，R25对应为 50.666% / 62.992%。准确率指p≥0.5对response outcome的分类正确率；Brier才是主要概率质量指标。

学生局部追加四维 ALC=0.211345789，相对R25 Δ=-0.000001632，95% CI=[-5.246056892522765e-06, 1.2121310454350758e-06]；B0/B1逐元素保持R25。仅需求局部模型 ALC=0.212701057，更弱。数值变动极小，不作有实质高预算收益的主张。

完整六预算、BM-equal、三个pool、BM切片、accuracy、区间详见 [demand_pilot_tables.md](demand_pilot_tables.md) 和CSV。源数据被多轮使用，这些都是探索性结果。

## 机制理解与失败边界

1. 合法的需求标签不自动产生稳定的成功率关系。本轮教师分数显著恶化，说明这个小样本、四维、线性先验的映射在新BM上不可靠。可能原因包括跨BM成功率方向变化、小样本不稳定、遗漏视觉/上下文、rubric质量和表示尺度；没有额外实验来唯一识别原因，不能宣称ADeLe总体无效。
2. 当前学生几乎只能达到常数grade均值。平均MAE下降不到0.01级，置换学生也接近；不能说已经获得高保真需求提取器。每rubric/每BM误差见grade_group_metrics.csv。
3. 学生是已有MiniLM/metadata的确定性映射。运行时没有独立新增raw information；小幅先验改善可以来自教师监督诱导的正则化/特征重加权。教师本身未通过独立信息对照，不能把学生轻微收益解释为已证实“认知需求结构”。
4. 需求teacher可看完整规范化文本，而MiniLM输入实际只留最多126内容tokens（前96+后30）加2特殊tokens；canonical输入还受20,000字符上限。原始长文本中丢失的信息无法由这个学生恢复。长度统计由R73最终质量审计提供，不能用模型默认256token替代真实代码窗口。
5. 教师的BM内联合置换只有68.5%完整items改变；学生逐维置换31.2%有效grades改变。同BM同等级项不改变，控制保留BM级分布；它检验的是实例对应关系，不是抹除全部任务类型信息。

## 回归与停止决定

prior_student 相对R25最差BM：cmmlu/formatted_tables Δ=+0.008370484；>=.005回归为 {'bbq/formatted_tables': 0.005162566026251203, 'cellverse/formatted_tables': 0.0052099936090877486, 'cmmlu/formatted_tables': 0.008370483543343965, 'faithcot/formatted_tables': 0.006456634982082682}。teacher对base最差BM：emoji_attack/formatted_tables Δ=+0.324155426。

冻结合同要求教师与部署候选共同通过来源进入条件。两个部署候选形式上通过小幅来源比较，但教师条件全部失败，故 `decision.json` 明确 **stop_at_pilot_keep_R25**。不运行外层、不扩大注释、不改ridge或维度寻优。No-Peer实际保留R25；本轮最强新增来源候选是prior_student，但没有晋级为新方案。

## 资源与工程产物

来源全部训练、预测、区间计算耗时 **32.52秒**，CPU线程2。四个学生head压缩文件共 **33941 bytes**。独立fresh-encoding测量（不写旧cache）：MiniLM+metadata+4head单item中位 **65.29ms**，8item batch中位 **310.01ms**；模型初始化 **1.966秒**，首次单item warmup额外 **5.560秒**，不包含在warm中位延迟中。已计算640向量后，1000item四head中位 **32.106ms**。包含/排除项、三次观测、固定8items见 `cpu_feature_benchmark.json`。这是本机微基准，不是Hosted吞吐承诺。

所有inner head与response prior以NumPy保存。`demand_student_cpu.py::load_demand_models` / `predict_demands` 可对新640向量推理；实际新item端到端入口为 `dual_track_cpu_features.CPUItemFeatures(threads=2).predict_demands(item, heads)`，只接收item_content/item_features、不查dataset和标签cache。8item实算特征与原cache最大误差7.45e-08，grade预测最大误差9.32e-08。本轮没有晋级，实际方案仍使用原R25。没有要求调用Luna才能完成学生推断。没有新模型下载、GPU或收费API。

## 重放与异常记录

保存全teacher/full-source probabilities、counts、successes、subject/item/variant identity、fit清单和代码snapshot。重放 **4,394,670** 个概率值，Brier/accuracy误差最大 **0**；历史R25重建误差≤2.22e-16。R25/R56/R66在原source全query上的分数复用核对误差≤0。

首次运行在任何fit前因R66 query metadata与variant数组顺序的错误断言中止；原 `contract/` 保留，解释见 `demand_pilot_IDENTITY_AMENDMENT.md`，修复后 `contract_v2/` 冻结。不是重调参数，也没有把失败运行当效果结果。历史资产未修改。
