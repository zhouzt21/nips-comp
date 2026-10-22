# PAIEC research workspace

PAIEC 概率预测竞赛的本地研究工程，包含官方 baseline 的原始快照与后续算法、评估和研究记录。

## 入口

- [工程与预测器说明](paiec_baseline/README.md)
- [Loop v3 当前发现与证据边界](reports/research_loop_v3_findings.md)
- [实验复现命令](reports/research_loop_v3_reproduction.md)
- [原始代码来源](paiec_baseline/UPSTREAM.md)
- [版本管理与原版恢复](doc/GIT_WORKFLOW.md)
- [实验记录的保存范围](paiec_baseline/runs/README.md)
- [测试入口](tests/README.md)
- [目录迁移与历史路径说明](doc/README.md)

## 目录职责

```text
nips-comp/
├── doc/                    研究计划、参考材料和工程说明
├── reports/                实验报告及报告图表
├── research_loop/          研究状态、协议、决策与文献记录
├── tests/                  项目测试
└── paiec_baseline/
    ├── mlp/                模型、特征、训练与评估核心
    │   ├── audit/          数据审计与失效诊断
    │   ├── run/            实验启动脚本
    │   ├── verify/         结果复核与回放
    │   ├── reporting/      汇总、绘图及报告生成
    │   ├── state/          研究状态更新
    │   └── data/           benchmark 划分工具
    ├── tools/              数据准备、打包、检查等工具
    ├── runs/               历史实验配置、结果及源码快照
    └── ...                 empirical_mean、smoothed_mean、BLE 等预测器
```

从工程根目录运行全部测试：

```powershell
python -m pytest
```

训练、实验和打包命令仍从`paiec_baseline/`执行；例如
`python -m mlp.audit.audit_evaluation_items`。具体实验依赖与运行条件见复现说明。

`paiec_baseline/` 是本仓库的普通目录，由根目录 Git 统一管理。无需初始化 submodule 或同步上游仓库。

原始版本固定在标签 **`paiec-original-cc88a95`**；`main` 保存后续修改。原始 baseline 的许可证位于 [LICENSE](paiec_baseline/LICENSE)。该许可证来自上游，本文件不替外部数据、预训练模型或参考材料重新声明许可。

公开数据、特征缓存、训练权重及提交 ZIP 不包含在 Git 中；克隆仓库不会自动恢复这些本地产物。实验配置、指标、源码快照和运行日志保留在`paiec_baseline/runs/`，报告与研究状态位于工程根目录。运行训练或重建历史实验前，请阅读对应的复现说明与依赖要求。
