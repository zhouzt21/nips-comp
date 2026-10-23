# plan_7 Starting Kit 获取与运行协议审计合同

日期：2026-10-23（UTC+8）。完整原始授权与要求保留在 `doc/plan_7.md`，不改写。类型：科研软件基础设施/运行协议审计，不开展新效果研究。

## 问题与验收

确认当前可访问的官方Starting Kit身份与版本、其相对GitHub的接口差异，以及它是否公开R72所需peer抽样规则。取得ZIP后检查大小/SHA256、CRC与路径安全、完整清单、关键源码行号；使用原生CLI与九类合成事件测试接口。只有真实Hosted事件标记verified_runtime；本地合成测试不提升为该状态。

## 访问与边界

检索用AnySearch，官方入口Codabench17828、aims官网和aims-foundations/paiec_baseline（含issue3）。不读取/导出浏览器凭据，不绕过登录或证书，不抓取隐藏数据。无可用登录会话时提供手动下载路径；无受支持API文档时不猜测私有下载URL。公开网页与GitHub源码获取可以继续。

## 存储与停止规则

新产物位于 `D:/nips-comp/paiec-official/starting-kit/`，仓库新报告/manifest保存引用，旧R1–R72代码、结果和fallback只读。各网络路径采用有界重试；同样权限/TLS失败不无限检索。不能取得ZIP时准确保留未完成状态，完成公开证据和恢复脚本后等待用户官方文件。取得客户端但缺coordinator时完成客户端审计，peer分布继续unknown。

## 可复核记录

URL、获取时间、版本/commit、hash、真实命令、stdout/stderr、合成fixture与测试结果分别保存。原始包不修改；测试副本独立。不运行新训练、不改变R72、不正式提交、不外发消息。
