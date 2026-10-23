# PAIEC Execution Prompt：下载、验证最新版 Codabench Starting Kit，并审计 Hosted Runtime Contract

## 一、任务角色与最终目标

你是一名 **Competition Infrastructure Engineer、Runtime Protocol Auditor 与 Reproducibility Engineer**。

当前 PAIEC 已完成 R1–R72。R72 pooled3 在三 peer 条件共享模拟中取得 0.190683911 Brier ALC，但尚未确认真实 Hosted coordinator 是否提供相同的跨 Subject 标签可见性。

本轮的任务是：

**从 PAIEC 官方 Codabench 下载当前账号可获得的最新 Starting Kit，恢复其真实运行时协议，比较现有 GitHub baseline，并确定它能够为 R72 的部署提供哪些新证据。**

本轮只做官方文件获取、版本审计、接口验证和必要的本地合成测试。

不训练新模型，不修改 R72 算法，不创建正式提交，也不访问或提取隐藏测试数据。

## 二、项目环境与官方入口

本地项目：

`D:\nips-comp\nips-comp`

基线目录：

`D:\nips-comp\nips-comp\paiec_baseline`

现有 PAIEC CPU runtime：

`D:\nips-comp\paiec-runtime-win64-py312-cpu\python.ps1`

官方来源：

- 竞赛官网：https://aims.stanford.edu/competition/
- Codabench：https://www.codabench.org/competitions/17828/
- GitHub baseline：https://github.com/aims-foundations/paiec_baseline
- 已知运行时版本差异：https://github.com/aims-foundations/paiec_baseline/issues/3

先确认这些地址仍为官方入口，不使用搜索结果中的非官方 ZIP 镜像。

## 三、Phase 1：实际获取最新版 Starting Kit

### 1. 首选：已登录的 Codabench 页面

进入 PAIEC 竞赛，确认登录账号具有参赛访问权限。

寻找官方提供的 Starting Kit 下载入口，可能位于 Files、Get Started、My Submissions 或竞赛资源区域，以实际页面为准。

如果存在多个 Kit 或不同阶段版本：

- 阅读文件说明、发布日期、阶段及版本标识；
- 优先选择当前适用阶段的官方最新版本；
- 不因文件名包含 latest 就直接认定最新；
- 记录全部候选文件的可见元数据及选择理由。

实际下载 ZIP，而不是只保存网页链接。

### 2. 登录或下载受阻时

按以下顺序处理：

1. 检查是否需要参赛注册或组织者批准。
2. 检查官方浏览器登录下载是否可用。
3. 如果有经过授权、文档明确支持的下载 API，才考虑通过 API 下载。
4. 如果遇到 TLS EOF、超时、403 或重定向登录，记录错误和实际发生的步骤。
5. 不关闭证书校验、不绕过身份验证、不使用私人或未经授权的文件地址。

若当前 AI 环境没有登录浏览器或图形操作能力，向用户提供**准确的手动下载步骤及保存位置**；用户完成下载后继续执行本地审计。

不得声称已成功下载实际上未取得的 ZIP。

### 3. 原始文件保存

建立独立目录，例如：

`D:\nips-comp\paiec-official\starting-kit\`

原始 ZIP 保持不变，另外建立解压副本。

记录：

- 官方下载页面及来源；
- 获取时间；
- 文件名、文件大小；
- SHA256；
- 文件内版本标识；
- 是否来自已认证的 Codabench；
- 是否确实是当前可见的最新版本。

不得覆盖项目现有 GitHub 文件、R1–R72 代码或历史实验资产。

## 四、Phase 2：Starting Kit 完整性与版本审计

解压后先列出完整文件清单，阅读 README、运行说明及版本文档。

重点查找：

- `streaming_ingestion.py`
- `check_submission_zip.py`
- `run_smoke_test.py`
- `competition_contract`
- ingestion / scoring / worker 相关程序
- coordinator 或 sampling 相关代码
- synthetic input、example evaluation event、test fixtures

实际文件名和位置以新 ZIP 为准。

### 对照 GitHub Issue #3

重点确认新版是否包含：

- `PIPELINE_RUNTIME_VERSION`
- `PAIEC_STREAM_ROLE`
- `configure`
- `acquisition_finished`
- `evaluation_finished`
- `prediction_workers`
- `competition_contract`

分别记录：

1. 在新 Kit 中是否存在；
2. 对应源文件及代码位置；
3. 实际输入输出、时序和错误处理；
4. 与本地 GitHub baseline 的差异；
5. 哪些变动影响 `predict(input, labeled)`；
6. 哪些仅影响运行部署、并行 worker 或 checkpoint 生命周期。

同时比较关键文件的 hash、接口签名和协议逻辑。

必须区分“GitHub 与 Kit 内容不同”和“Kit 已经揭示 coordinator 的抽样行为”，不能把前者当成后者的证明。

## 五、Phase 3：重点审计 R72 所依赖的共享输入合同

围绕下面的问题检查实际源码、配置和测试样例。

### A. `labeled` 是否为跨 Subject 共享快照？

确定：

- `labeled` 从哪里构建；
- 它由哪个进程或事件提供；
- 何时传递给预测 worker；
- 多个目标预测是否共享同一份证据；
- 是否按 target Subject 或 Benchmark 过滤；
- 当前调用是否可访问其他 Subject 的标签。

如果客户端只是原样转发 coordinator 给出的 `labeled`，而 coordinator 代码未公开，则将其明确记为“转发机制已确认，抽样内容未知”。

### B. Peer 数量和 Subject 身份

检查是否存在明确规定：

- 每个 Benchmark 抽取多少 Subject；
- 目标 Subject 能看到多少其他 Subject；
- 多个 Subject 的配置字典能否可靠区分；
- 相同可见配置是否可能对应不同匿名 Subject；
- 是否保证至少一个或三个 peer。

特别注意：**R72 pooled3 的三 peer 是先前实验的条件假设，不是当前已经核实的 Hosted 事实。**

### C. Item 匹配与 acquisition 分割

确定：

- 是否有稳定的 item identity；
- 相同正文、features、interactors 是否保证是同一潜在 item；
- 不同 Subject 的 acquisition/evaluation 划分是否共享；
- 是否允许 peer acquisition 与 target query 同输入；
- 重复 response 如何传递。

没有明文保证时，不能自行补充。

### D. 调用生命周期

检查：

- worker 是否会重启；
- checkpoint 是否重新构建进程；
- 是否允许依赖跨调用缓存；
- `predict()` 是否应完全根据本次输入确定结果；
- B0 是否为全局空 `labeled`；
- 每预算证据的嵌套关系；
- 输入规模、worker 数量、内存和超时限制。

对每项提供源文件、具体函数及证据位置。

采用以下状态：

`verified_runtime`、`verified_official_source`、`documented_only`、`inferred`、`unknown`、`unsupported`。

只有真实 Hosted 执行事件才能标记为 `verified_runtime`。

## 六、Phase 4：本地协议测试

优先使用下载 Kit 自带的检查和 smoke-test 程序。

运行前确认实际 CLI 参数，不猜测参数名称。

通过合成输入测试：

- B0 无标签；
- B1 只有 own 标签；
- 有 1 个 peer；
- 有 3 个 peer；
- 不同 Subject 共享 `benchmark_id`；
- 相同可见 Subject 配置；
- 同正文但 metadata 不同；
- 没有可匹配同输入 peer；
- worker 重建后独立调用。

以简单的有限概率预测器或 R25 的本地兼容适配器进行测试，不需要重新训练。

如果条件允许，可对 R72 冻结实现进行纯合成协议测试，确认无 peer 时回退 R25，以及存在 peer 时能够解析输入。

这些测试只能验证**接口兼容性与实现稳定性**，不能证明真实 Hosted 的 peer 抽样分布。

不得用隐藏提交、评分反馈或输出概率编码来反向提取私人测试数据。

## 七、Phase 5：实际回答最关键的部署问题

完成文件与代码审计后，形成以下判断：

**Q1：取得的是不是当前官方可获取的最新 Starting Kit？**

给出来源、版本、SHA256 和依据。不能确认最新时说明原因。

**Q2：相比现有 GitHub baseline，运行时协议发生了什么变化？**

特别指出任何会造成当前提交包在 Hosted 下失败的差异。

**Q3：R72 能否合法读取其他 Subject 的当前标签？**

区分官方允许、多 pair 证据转发已确认，以及实际 peer 组成仍然未知这三个层次。

**Q4：能否根据 Kit 确定三 peer 的实际分布？**

若不能，明确定位是哪段 coordinator 行为仍不公开，不继续无目的检索。

**Q5：现在能否构建与真实运行时兼容的 R25/R72 提交包？**

分别判断接口兼容性、输入可用性和推理资源。

**Q6：如果还有未知问题，需要组织者补充什么最小信息？**

输出简短、准确的英文问题清单，优先询问：

- `labeled` 是否为跨 sampled pairs 的共享快照；
- 同 Benchmark 是否保证可见多个 Subject；
- peer 数量与组成是否有固定规则；
- 是否有真实或合成的 coordinator evaluation-event 示例；
- peer acquisition / evaluation 是否存在规定的跨 Subject 重叠关系。

不索取隐藏任务内容或标签。

## 八、结果保存

建议在项目中生成：

`reports/paiec_starting_kit_download_audit.md`

`research_loop/PAIEC_STARTING_KIT_VERSION_MANIFEST.json`

`research_loop/PAIEC_RUNTIME_CONTRACT_UPDATE.md`

`research_loop/PAIEC_STARTING_KIT_DIFF.md`

原始 ZIP、解压文件、diff 和测试日志保存于：

`D:\nips-comp\paiec-official\starting-kit\`

沿用现有项目的证据登记规范；不覆盖 R69/R72 的原始运行时审计，改为提供新版本和引用关系。

如果项目现有路径有所不同，先检查再决定，不凭空声称文件已经生成。

## 九、完成标准与执行要求

任务分为：

**下载成功 → ZIP 完整性确认 → 版本差异审计 → 输入合同审计 → 本地测试 → 最终结论。**

如果下载被账号权限阻挡，报告实际阻挡点，给出用户可以直接执行的步骤，同时完成公开 GitHub 与官网能够支持的部分审计。

如果成功下载，但没有私有 coordinator，也算完成 Starting Kit 获取任务；须将抽样合同明确保留为未解决问题。

不进行：

- 新模型训练；
- R72 超参数搜索；
- 正式提交；
- 自动发送组织者邮件；
- 任何隐藏测试数据提取；
- 对 R1–R72 原代码和 fallback 的覆盖。

**现在开始执行：首先从官方 Codabench 获取 Starting Kit，验证 ZIP 与 SHA256；随后检查其与 GitHub baseline 的版本差异，重点恢复 `labeled` 的转发、来源与可见性合同。最终给出可核查的文件、证据位置、已确认事实和仍需组织者澄清的最少问题。**