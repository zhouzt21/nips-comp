# R53：ADeLe 标注输入准备与资源可行性

日期：2026-10-22。状态：**dry-run 已完成；用户已确认 ADeLe 等待资源，先完成 CNP/IRT。** 当前没有标注结果、模型训练、PAIEC 正确率/Brier/ALC 或 API 调用。128 × 4 是输入与资源可行性样本，不是正式 PAIEC 效果实验，也不表示 API 授权。

## 本次已完成

从固定 PAIEC public corpus revision `54074453da7b9911be0f48e4c560be547bc84d21` 的 outer fold 0 training benchmarks 中选取 128 个不同内容 items。采样只使用既有 seed 73、每 benchmark 至多 25 个 subject episodes 的 query item IDs，按 benchmark 名称排序轮转，并按 `SemanticFeatures.canonical` 文本 SHA256 全局去重。最终覆盖 25 个 training benchmarks；前三个各 6 题，其余各 5 题，未从 outer heldout benchmarks 取样。

来源缓存 fingerprint：`a739dcd55115c0bbb46e1487b917e75548b86d63d46fa91f2251fa85668af453`。源候选有 12,109 个 benchmark 内 distinct query IDs。保守排除包含结构化 assistant role 的 384 个候选；轮转时跳过 21 个已经入选的相同 canonical 文本。该选择用于检查标注工程成本，不主张代表完整比赛分布。

四个维度固定为作者的 `AS`（Attention and Scan）、`MCr`（Identifying Relevant Information）、`QLl`（Logical Reasoning）、`KNf`（Knowledge of Formal Sciences），每个 item 构造四个请求，共 **512 个唯一请求**。

官方来源固定为 `Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation` commit `816321f327dfe66910b72b514156aaa296b15f00`。脚本检查 commit 和源文件相对 HEAD 无差异，再直接加载作者的：

- `src/adele/annotation/prompts.py:16`：`build_annotation_prompt`。
- `src/adele/annotation/prompts.py:62`：`build_batch_request`。
- `src/adele/rubrics/catalog.py`：作者维度名称与 rubric 加载方式。
- `src/adele/rubrics/data_v1/{AS,MCr,QLl,KNf}.txt`：完整原始 rubric 与等级实例。

没有复制后改写 prompt。请求维持作者的单个 `user` message、`temperature=0`、`max_completion_tokens=1000` 结构。模型名称明确为 `MODEL_SNAPSHOT_PENDING__DRY_RUN_ONLY`；manifest 的 `model_snapshot=null`。这是不能直接作为真实模型标识使用的占位符，实际 snapshot、供应商与预算仍待资源就绪后确定。

## 输入和证据边界

脚本只解码 npz 的 `metadata` 成员，从中取 benchmark、seed、support/query item IDs；没有解码 `labels`、`counts`、`successes` 等数组。Parquet 只投影 `item_id`、`content` 两列，未打开 `response.parquet`，未加载 subjects 或 item_features。为记录文件身份所做的 SHA256 是原文件字节摘要，不用于读取或选择标签值。

为避免把结构化答案或评测输出作为 task，脚本整题排除 `gold`、`groundtruth`、`label`、`target`、`correct_answer`、`solution`、`response`、hidden/private 字段与 assistant role；不会删除这些字段后重新组合输入。当前实际排除原因只有 assistant role。其余输入直接使用现有 `SemanticFeatures.canonical`。这与 MiniLM 的公开文本抽取规则一致，但尚未施加 MiniLM 后续 126-token 首尾窗口：需求 judge 接收 canonical 文本，最长 20,000 字符。此差异必须在将来的效果实验中披露，不能把它写成与 R25 完全相同的保留 token 预算。

没有读取或复用旧 ADeLe battery 的问题、需求标签、subject 响应。benchmark 名称和 item IDs 仅作为请求身份与审计信息，不拼入 judge 的 task 文本。当前结构化字段排除并不是自然语言中所有潜在答案的语义判定；输入范围以比赛提供的公开 `content` 为边界。

## 实际输出与结构核对

外部目录：`D:/nips-comp/paiec-dataset/reproduction_v4/adele_pilot/`。

| 文件 | 内容 |
|---|---|
| `requests.jsonl` | 512 个作者格式的待标注请求，约 4.64 MB；模型仍为占位符。 |
| `items.json` | 128 个 item 的来源、query episode index、raw content/canonical 文本 hash、字符数。 |
| `request_statistics.json` | 每请求的 prompt hash、输入字符数、UTF-8 字节数。 |
| `manifest.json` | dataset revision、outer split、全部源 cache/Parquet/代码/rubric hashes、列名、采样规则、统计及检查结果。 |

实际输入字符总数 **4,413,536**，每请求最少 3,214、最多 29,223、平均 8,620.1875。**这些是字符数，不是 token 数、费用报价或实际账单。** 准备阶段记录 elapsed 约 5.82 秒；这不是模型推理时间，也不是用户投入时间。

请求文件 SHA256：`573243e3a4cd89a7dbbb7813abf33d418702e684318c044c652b7b221a6d1584`。

结构核对通过：128 个 distinct canonical hashes；512 个唯一 request IDs；每维 128 个请求；仅 fold 0 training benchmarks；每请求只有作者的单个 user message 和 pending 模型标识。未执行标注、未计算模型效果、未下载模型。

## 复现命令与停止条件

实现文件：`paiec_baseline/mlp/run/prepare_adele_pilot.py`。在 `D:/nips-comp/nips-comp/paiec_baseline` 运行：

```powershell
& 'D:\nips-comp\paiec-runtime-win64-py312-cpu\runtime\python.exe' -m mlp.run.prepare_adele_pilot --output 'D:\nips-comp\paiec-dataset\reproduction_v4\adele_pilot_repeat'
```

输出目录须不存在或为空，脚本拒绝覆盖既有产物；也不包含 API 调用入口。复现后的时间戳和 elapsed 会变化，固定 inputs 的请求及 item hashes 应一致。官方 catalog 跳过 `UG_choice_num` 的提示来自作者校验器，因为 UG 不属于 18 个 0–5 rubric；本次四个维度均正常加载。

本轮在输入准备处停止，遵循用户“先完成 CNP/IRT，ADeLe 等待资源”的选择。后续只有真实 annotator snapshot、许可输入、明确预算与部署方式可用，且小样本解析、成本和延迟检查通过，才可冻结扩大标注或 PAIEC 效果实验合同。当前可复核的结论是输入已准备；不能据此声称 ADeLe 特征已生成、原论文效果已复现或比赛得分已提升。

原文和方法解释入口：`research_loop/literature/reproduction_v4/adele.md`、`adele_sources.json`。
