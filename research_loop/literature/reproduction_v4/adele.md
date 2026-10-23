# ADeLe / General Scales：原文、官方实现与 PAIEC 适用边界

核对日期：2026-10-22。状态：`sources_verified; conditional_route; annotation_runtime_unavailable`。本轮仅核对来源和部署可行性，没有标注新样本、调用模型 API、训练 assessor 或报告 PAIEC 分数。

## 核对结论

用户方案中的论文身份正确：Lexin Zhou et al., **General scales unlock AI evaluation with explanatory and predictive power**, **Nature 652, 58–67 (2026)**，2026-04-01 在线发表，DOI [10.1038/s41586-026-10303-2](https://www.nature.com/articles/s41586-026-10303-2)。2025 年的 [arXiv:2503.06378](https://arxiv.org/abs/2503.06378) 是原预印本入口，本次方法依据为 Nature 正式全文及其补充材料。

其核心特征由通用生成式 LLM 按完整 DeLeAn rubric 逐题逐维度打分。当前离线 MiniLM 是文本 embedding 模型，官方仓库没有把 MiniLM embedding 直接转换为这 18 个等级的已训练模型。因此，现在可以忠实取得 rubric、构造标注输入、复用 CPU 随机森林；**不能用当前 MiniLM 直接完成忠实的 ADeLe 需求编码**。用手写关键词、字数阈值或 embedding 对 rubric 的余弦相似度替代等级评分，均是新启发式，不能记为 ADeLe 复现。

这是用户方案的条件后续路线，是否进入实验仍取决于 CNP/ANP、IRT 的结果以及标注和推理条件。暂不扩大为第三条训练任务。

## 原文与作者代码的对应关系

官方项目页 [ADeLe](https://kinds-of-intelligence-cfi.github.io/ADELE/) 直接链接 [Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation](https://github.com/Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation)。固定 commit：`816321f327dfe66910b72b514156aaa296b15f00`，提交日期 2026-10-12。该版本已包含后来统一的 Python 工具包，同时保留原始 `predictive_power/` notebook；不能把统一工具包的默认值自动当作论文当年的全部配置。

| 原文定位 | 作者代码定位（相对固定仓库） | 已核对事实 |
|---|---|---|
| Methods，LLM annotators and inter-rater analysis；主文 PDF 第 13 页；补充 Table 23 | `src/adele/annotation/prompts.py:16`、`:62`；`src/adele/annotation/annotator.py:78` | 每个 item × rubric 独立标注。原文使用 Azure 的 `gpt-4o-0513`，temperature 0，输出最多 1,000 tokens。工具包默认 `gpt-4o`，可选 OpenAI Batch 或 litellm direct；需记录实际 snapshot，不能把别名视为同一模型。 |
| Methods，General scales；补充 Section 2 | `src/adele/rubrics/data_v1/`；`src/adele/rubrics/catalog.py:31` | 18 个 0–5 需求维度：11 个认知维度、5 个知识维度、AT 与 VO。UG 另为答案形式的控制变量。rubric 包含各等级描述与实例锚点。 |
| Methods，Dissecting the demand–ability space；Assessors and metrics；PDF 第 13–15 页 | `predictive_power/predictive_power_stats_rf-assessor.ipynb`；`src/adele/analysis/prediction.py:184` | 每个 subject 单独以 18 个需求 + UG 预测其成功概率。原文 RF 调过 `min_samples_split ∈ {2,50,200}`；保留 notebook 和新 API 的复现默认是 100 棵树、50、entropy。 |
| 主文 Predictive power…；Extended Data Tables 2–4 | 原 notebook OOD 函数；`src/adele/analysis/prediction.py:117` | ID 按实例划分；OOD 按 task 或 benchmark 唯一组划分。10 折、10 seeds，主要报告 AUROC/ECE；这不是 PAIEC 的六预算 pair-equal Brier ALC。 |
| 补充 Section 1.8，Tables 11–13；补充 PDF 第 26–29 页 | annotation API 支持换 judge | 原文也验证 DeepSeek-V3 和 LLaMA-3.1-8B-Instruct。与 Delphi 人类共识的平均 rWG 分别 0.83、0.74，GPT-4o 为 0.86。小生成模型方案有原文依据，但质量下降且不是现成 MiniLM 编码头。 |
| Unguessability 定义 | `src/adele/annotation/parsing.py:83` | 已知 n 个选择时 UG = 100 × (1 − 1/n)，开放题为 100；不能猜测缺失的答案形式。 |

需求标注器只接收 task 文本和 rubric；作者实现从 `row['prompt']` 构造请求，并不把 subject 的响应、正确性或 gold answer 放入需求 prompt。需求描述 item；subject correctness 用于后续 assessor 的监督。

VO 的原义是充分胜任的人阅读并完成任务所需的时间等级，包含完成时间。它不等于 prompt 字数。QLl 区分推理链、前提与逻辑构造复杂性；知识维度按所需知识水平区分。只用长度或关键词会改变其测量对象。

## 当前机器能做什么

- 原始 rubric、prompt builder、UG 计算和 sklearn RF 可以在 CPU 上运行。没有必要为阅读/核对安装整个 `adele[all]`。
- 原仓库无 `.pt`、`.bin`、`.safetensors` 需求编码器 checkpoint；核对范围是上述固定 commit。公开的预标注 battery 是数据，不是新题编码器。
- 论文的低成本 RF 训练数字不包含先前的 LLM 标注成本。固定 (N) 个 unique items、(d) 个维度时，官方调用数量是 (N d)，还需重复输入完整 rubric 和题目。
- 现有 MiniLM 可用于独立的 embedding 基线，也可在未来取得合法的需求标签后蒸馏一个编码头。后者是需要单独验证的 PAIEC 适配，不是作者已提供的预训练模型。
- LLaMA-3.1-8B-Instruct 是作者确实分析过的可选 annotator，原则上可部署，但本地目前没有其权重、推理配置、许可证接入和时间测量。本轮没有下载或尝试运行。不能从“可开放获取”推断当前 16 GiB 机器能在竞赛时限内完成逐题逐维度生成。

## 数据、许可与迁移限制

仓库根代码和附带文档为 MIT（须保留 copyright/license）；`ADeLe_battery_data/LICENSE` 将其数据标为 CC BY-SA 4.0，另需尊重原始 benchmark 条款。仓库 checkout 包含公开的 battery 和 15 个 subject CSV，本轮只查看 battery 列名、行数与 benchmark 名称用于辨认资产，没有以其训练或评价 PAIEC 模型。CSV 共 16,108 个 items；此数字是源论文 battery 规模，不是 PAIEC 新实验规模。

作者 README 也提供需要授权的 Hugging Face battery 入口；本轮没有访问授权凭据或绕过访问控制。通过公开 Git 仓库取得的源数据也不自动成为竞赛允许训练来源。PAIEC 适配只应使用当前比赛 inventory 允许且已对齐 split 的源数据。整体搬入 ADeLe battery 或借其 subject 响应训练会违反本任务的数据范围；本轮未这样做。

原论文训练的是每个已有 subject 的 assessor，没有直接解决一个没有历史响应的新 subject 如何零标签获取完整 ability profile。把需求特征接入 IRT 或 CNP 是可检验的新迁移假设，需要现有 subject 冷启动协议，不能继承论文在已见 subject 上的 OOD 数字作为保证。

## 条件满足后的最小实验（尚未执行，需先冻结）

第一步是标注可行性小样本，而非完整训练。仅从已允许的 PAIEC outer-training 来源选择 128 个 distinct items，覆盖多个 benchmark；预先固定 `AS, MCr, QLl, KNf` 四个作者维度，使用原 rubric 和原 prompt。按官方实现需要 512 次 judge 调用；这只是调用数量，不是已获成本/时限测量。标注输入只包含推理时可见的文本/选项，不含正确标签、subject 响应或不可见 metadata。保存 item 内容 hash、rubric hash、model snapshot、解析成功率、等级分布、耗时和 provider 用量。若拟改选维度，应在看分数前另写冻结理由。

上述四维只是为控制费用的建议性 ablation，不能声称已复现作者 18 维的效果。全量 18 维、其他 judge 或更小模型均是不同配置。没有 approved provider/可用本地 judge 时，这一步停在输入构造；不以词表伪造标签。

如果小样本显示标注能稳定产生，并且隐藏推理可使用相同机制，再将选定维度扩展到固定合法训练集合和对应 held-out items。CPU 下复用 RF 100 棵树、`min_samples_split=50`、entropy 作为需求特征 assessor，对照相同信息的 MiniLM RF。独立比较需求、MiniLM、两者拼接，并始终报告冻结 smoothed mean 与 R25。所有外层 item/benchmark 隔离、重复 responses、pair 等权和六预算规则由 PAIEC 现有 evaluator 实施；不能直接把作者 notebook 的 pooled instance metric 当成比赛分数。

如果目标是为新 subject 构造 B0 先验，必须另外声明 subject 条件化的来源与冷启动处理。原文的 subject-specific RF 只能在该 subject 有合法训练响应时直接适用。需求特征只有在独立对照显示增量后才能接入 CNP/IRT；不同时调整标注维度、judge、融合系数并用同一 outer 结果选最好者。

## source_gap / execution_gap

| 标识 | 缺口 | 影响与解决条件 |
|---|---|---|
| `ADELE-G1` | 当前环境没有与作者协议相符的新题需求 annotator；MiniLM 不具备该能力 | 暂不产生 ADeLe 需求特征或分数。需要 permitted judge API 或可打包的生成模型及确定版本。 |
| `ADELE-G2` | judge 的竞赛供应商支持、实际费用、每 item 延迟/资源尚未测量 | 保留条件路线；不能把 CPU RF 的 4 秒训练宣称为端到端效率。 |
| `ADELE-G3` | 预标注 ADeLe items 与 PAIEC inventory/split 的逐项许可、来源和内容对齐未完成 | 本轮不使用源 battery 训练；默认只在已批准 PAIEC 来源上重新标注。 |
| `ADELE-G4` | 官方没有现成的轻量本地需求 checkpoint；MiniLM 蒸馏需要合法 labels 与误差评估 | 无法立即替换 judge。不能把人工关键词或 cosine 分数记为原方法。 |
| `ADELE-G5` | subject-specific assessor 对完全新 subject 的迁移不由原文直接支持 | 若融入 PAIEC 零标签先验，另立 subject 模型和受控实验合同。 |
| `ADELE-G6` | 原文与当前工具包使用的 GPT-4o 标识不同，原 snapshot 可用性未确认 | 记录真实 judge 版本；替代 snapshot 记为受控适配，不能称逐参数复现。 |

## 原始证据存储

源资产统一位于 `D:/nips-comp/paiec-dataset/reproduction_sources/adele/`：

- `nature_paper.pdf` / `nature_paper.txt`：正式主文完整 PDF 和 pdftotext 文本。
- `nature_supplementary.pdf` / `nature_supplementary.txt`：补充全文。
- `nature_full.html`：正式页面原始 HTML。
- `nature_anysearch.md`、`project_anysearch.md`：AnySearch extract 原始输出。服务对长页内容有截断，不能替代完整 PDF。
- `upstream/`：固定官方仓库；`predictive_power_notebook_source.txt` 为其 notebook 的代码/Markdown 抽取。
- 本目录 `adele_sources.json` 记录 URL、commit、文件 hash、检索 request ID 和结论边界。

对下一轮的认识：这条路线的关键投入在“能否对新题一致地测量需求”，而 RF 是已有需求向量后的廉价学习器。当前表示不足即使成立，也不能直接推出需求特征有效；需要同时检验其标注可靠性、跨 benchmark 增量和可部署性。
