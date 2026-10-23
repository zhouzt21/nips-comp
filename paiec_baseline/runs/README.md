# 保存的实验证据

## 2026-10-22 plan_4历史正面信号补证

`research_loop/R62/`、`R63/`及`R64/full/`均保存完整5242pair×3seed的概率、逐pair指标、Brier/正确率、冻结合同和重计分审计。R64先完成来源pilot并通过预冻结条件再进入外层；其顶层summary保留原进入决定，最终结果看full/summary。新增fit10次、复用R43权重3份、无新embedding。R56_segment探索ALC0.193805399，但BM/一折/CI未通过；R55 query仅微小优势，R64模板平均仍略差R25。无晋级、无提交、fallback不变。主报告：`../../reports/historical_positive_followup_findings.md`，含机制理解；完整六预算正确率见`../../reports/historical_positive_tables.md`。本轮保留了性能中断产物与数值等价续跑记录，不应重复当成独立实验。

## 2026-10-22 plan_3低预算先验组合

`research_loop/R59/`保存R54/R56冻结权重及R25的同响应预测回放、四项Brier分解、预算重组、可靠性信号和oracle；`R60/`保存评分前冻结的单一B1首标签门控；`R61/`保存候选预测、配对结果和数值核对。824次来源池pair记录对应412不同pair，新增训练0。候选1/3池满足数值条件，整体未通过，完整外层condition_not_met；不修改R25/fallback。主报告含正确率和机制理解：`../../reports/low_budget_prior_composition_findings.md`。预测NPZ/权重快照为本地文件；需要完整保存而非只克隆Git。

## 2026-10-22 plan_2五候选

`research_loop/training_mechanism_v5/R54`–`R58`保存完整轨迹、支持集重采样、Benchmark风险、条件平均与初始化诊断。36次fit、996个epoch checkpoint；所有执行方案未通过全来源准入，MLP平均和初始化未触发。主报告含六预算Brier/ALC/预测正确率与机制分析：`../../reports/training_mechanism_v5_findings.md`。运行根保留环境、来源快照、stage锁、输入hash、数值/匹配审计、失败尝试和权重清单；NPZ为本地文件且Git忽略。没有新外层成绩或提交。原prompt与旧轮次保留。

## 2026-10-22 原文复现阶段

新增 `research_loop/R51/`（CNP/AttentiveCNP）和 `R52/`（amortized IRT）：18次真实来源内拟合完成，全部进入条件失败，没有外层新成绩或组合。报告与六预算Brier/ALC/分类正确率见 `../../reports/reproduction_v4_findings.md`。`research_loop/reproduction_v4/` 保存基线回放、28项检查日志、环境与本地权重SHA256清单。`.pt` 权重保存在本机各run目录，Git忽略；源码快照/指标不等于权重已经随仓库分发。

ADeLe仅完成源核对和512个待标注请求准备，用户确认等待资源；外部输入在 `D:/nips-comp/paiec-dataset/reproduction_v4/adele_pilot/`，不作为模型效果结果。

本目录保留原来的实验路径，方便报告、配置和代码互相引用。

Git保存：JSON/JSONL配置与指标、Markdown说明、Python历史源码快照、文本核验记录及运行日志。历史失败和来源内未通过的实验也保留；不能把文件存在解释为成功或独立验证。

Git不保存：逐样本CSV大表、NPZ权重/缓存、ZIP及本目录内生成的图片/PDF。报告中的必要图表保存在`../../reports/figures/`。原始公开数据与特征缓存位于项目外的D盘目录，未复制进仓库。

**复现边界：** 克隆得到的是代码和研究证据，不是完整运行环境或全部中间产物。部分后续诊断读取前序实验的CSV/缓存；需要按`../../reports/research_loop_v3_reproduction.md`的依赖顺序重建这些产物，或从保留的完整本地备份恢复。已有JSON与源码哈希有助于核对，不替代缺失的权重、数据和缓存。不要把本地绝对路径误认为其他机器已经具备的资源。

新增配置/指标会被Git纳入候选跟踪范围；提交前仍需检查`git diff --cached`，确保没有凭据或未经允许的原始数据。
