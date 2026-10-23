# PAIEC Starting Kit 审计进度（按用户要求暂停）

2026-10-23（UTC+8）。原始授权：[plan_7.md](../doc/plan_7.md)。**未取得官方Starting Kit ZIP，未确认最新版本，也未完成Kit/Hosted兼容性验收。**

用户提供 [Codabench issue #2679](https://github.com/codalab/codabench/issues/2679) 作为服务中断线索，并指示“所以先不管这件事”。未继续联网核验该issue；已暂停，不轮询或自动重启下载。

## 已完成与保留

- 官网原文仍指向Codabench17828。AnySearch提取失败，直接HTTPS请求发生TLS EOF，Edge返回ERR_CONNECTION_CLOSED，错误截图已保存。未到登录/权限检查，账号权限仍未知。
- 官方GitHub commit `cc88a959c1e5e7462f31a9215baba066ce6d6e2b` 的40个文本文件和Issue #3已取得。Issue #3为社区报告、open、0评论，不能代替维护者确认或实际Kit源码。
- GitHub streaming client、ZIP checker与本地对应文件hash相同；Issue #3提到的七个新版符号在这些GitHub实现中缺失。真实Kit差异仍待核实。
- 公开客户端16个合成协议用例完成。exchange/group_inputs/prediction_stream为本地注入替身，不是实际coordinator或worker实现。
- 原生GitHub ZIP checker通过。BLE `tools/smoke_test.py --mock`未整体通过：本地代理测试预期TLS错误摘要，实际network错误摘要，断言失败；完整日志保留，未修改官方代码掩盖失败。
- 两个独立进程运行官方empirical_mean结果一致；R72冻结数值核心0/1/3peer检查通过。后者未接入真实编码器或raw接口适配器，不等价于提交包验收。

## 存储

独立证据根目录：`D:/nips-comp/paiec-official/starting-kit/`。

|内容|根目录下位置|
|---|---|
|AnySearch原文与获取记录|`retrieval/`|
|官方GitHub快照|`github/cc88a959c1e5e7462f31a9215baba066ce6d6e2b/`|
|hash、行号、本地diff|`audit/source_comparison.json`及同目录文件|
|访问失败记录|`audit/access_attempts.json`|
|浏览器错误截图|`output/playwright/codabench-connection-closed.png`|
|合成输入与结果|`tests/summary.json`、`tests/fixtures/`、`tests/protocol/`|
|原生命令和错误日志|`tests/native_github_*.json`、`.stdout.txt`、`.stderr.txt`|

版本清单：[PAIEC_STARTING_KIT_VERSION_MANIFEST.json](../research_loop/PAIEC_STARTING_KIT_VERSION_MANIFEST.json)，官方Kit字段仍为空。

`tests/synthetic_probe_DO_NOT_SUBMIT.zip`仅为自建常数预测器检查样本，不是官方Starting Kit或正式提交包。`originals/`尚无官方ZIP。

## 恢复入口

用户恢复事项后，先取得官方ZIP及可见版本/阶段说明，保存到 `D:/nips-comp/paiec-official/starting-kit/originals/`。[inspect_starting_kit_zip.py](../research_loop/inspect_starting_kit_zip.py)已准备，可独立保留原包、检查CRC/解压路径并生成清单；来源与“最新”仍须页面证据。

尚待：下载与版本证明、真实Kit diff、完整合同更新、Kit原生smoke和部署资源结论。当前无真实Hosted事件、无训练、无提交、无外发；R1–R72原代码和fallback未修改。
