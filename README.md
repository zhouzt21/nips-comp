# PAIEC research workspace

PAIEC 概率预测竞赛的本地研究工程，包含官方 baseline 的原始快照与后续算法、评估和研究记录。

## 入口

- [工程与预测器说明](paiec_baseline/README.md)
- [Loop v3 当前发现与证据边界](paiec_baseline/reports/research_loop_v3_findings.md)
- [实验复现命令](paiec_baseline/reports/research_loop_v3_reproduction.md)
- [原始代码来源](paiec_baseline/UPSTREAM.md)
- [版本管理与原版恢复](GIT_WORKFLOW.md)
- [实验记录的保存范围](paiec_baseline/runs/README.md)

`paiec_baseline/` 是本仓库的普通目录，由根目录 Git 统一管理。无需初始化 submodule 或同步上游仓库。

原始版本固定在标签 **`paiec-original-cc88a95`**；`main` 保存后续修改。原始 baseline 的许可证位于 [LICENSE](paiec_baseline/LICENSE)。该许可证来自上游，本文件不替外部数据、预训练模型或参考材料重新声明许可。

公开数据、特征缓存、训练权重及提交 ZIP 不包含在 Git 中；克隆仓库不会自动恢复这些本地产物。实验配置、指标、源码快照、运行日志及研究报告保留在原路径。运行训练或重建历史实验前，请阅读对应的复现说明与依赖要求。
