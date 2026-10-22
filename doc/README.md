# 文档与历史路径

- 当前研究流程：[Loop v3](loop_2.md)。
- 实验报告：[reports/](../reports/)。
- 研究状态：[research_loop/](../research_loop/)。
- 版本管理：[GIT_WORKFLOW.md](GIT_WORKFLOW.md)。
- Python模块与源码路径迁移表：[module_path_migration.json](module_path_migration.json)。

## 目录迁移

| 历史位置 | 当前位置 |
| --- | --- |
| 根目录计划、Loop与四份参考材料 | `doc/` |
| `paiec_baseline/reports/` | `reports/` |
| `paiec_baseline/research_loop/` | `research_loop/` |
| `paiec_baseline/tools/test_*.py` | `tests/test_*.py` |
| `paiec_baseline/mlp/`中的辅助脚本 | `mlp/audit/`、`run/`、`verify/`、`reporting/`、`state/`、`data/` |

训练和实验命令的工作目录仍是`paiec_baseline/`；读取研究协议时使用
`../research_loop/`，输出报告使用`../reports/`，读取研究计划使用`../doc/`。
报告中的复现入口已更新为新的模块名。

## 历史证据

`paiec_baseline/runs/`中的指标、源码快照、缓存和日志保持原始内容。
事前协议`*_protocol.md`、来源哈希JSON和原始研究提示词也保留原文。
其中的旧路径代表当时的布局，按上表定位；不把目录调整解释为一次新的实验。
需要逐字节复核旧实验时，使用对应源码快照或迁移前版本，不能将当前重构后的
源码哈希归属于旧实验。
