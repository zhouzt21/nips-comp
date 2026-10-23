请在 `D:\nips-comp\nips-comp\paiec_baseline` 中完成 PAIEC 两套最终提交候选的独立打包与本地验收。

**候选 A：** 冻结 R25 No-Peer，使用原 R25 的完整语义、metadata、局部预测流程，不依赖其他 Subject 标签。

**候选 B：** 冻结 R76/R81 确认的 Partial8 Shared-Peer，实现严格的无 Peer/身份歧义回退 R25。不得加入 R79 门控或未经验证的新算法。

执行要求：

1. 阅读 R25、R76、R77、R81 的最终实现、冻结合同和运行时适配器，复用现有代码，不重新发明模型。
2. 为两套方案分别建立完全独立的提交目录及 ZIP，根目录包含官方要求的 `model.py`。
3. 检查 MiniLM 权重、所有 Python 依赖、绝对路径和缓存访问，确保 ZIP 在干净环境中能够独立推理。
4. 编写并实际执行接口、六预算、无 Peer、1/2/3 Peer、身份歧义、重复响应、独立进程与资源测试。
5. 使用冻结实验的输入和预测进行数值一致性核验；不能只检查代码可以导入。
6. 使用官方仓库现有的 ZIP 检查器和能够访问的最新本地 smoke-test 工具。记录未能完成的 Hosted 验证。
7. 输出两个 ZIP、SHA256、依赖清单、验收报告和可复现命令。
8. 不修改 R1–R81 的历史产物，不覆盖 `smoothed_mean_strength2.zip`，不重新训练，不正式提交 Codabench。

产物：

- `dist/paiec_r25_nopeer_final_candidate.zip`
- `dist/paiec_partial8_shared_final_candidate.zip`

如任一候选未通过完整验收，明确标记 `NOT_READY`，不能声称提交包已就绪。

**目标是完成两套可独立运行、可复现、可供最终选择的候选，而不是现在决定最终采用哪套。  然后把最新版仓库提交并推送一下**
