# 目录与模块重构记录

## 结构调整

- 根目录`doc/`沿用用户整理的计划与参考材料。
- `paiec_baseline/reports/`移至根目录`reports/`。
- `paiec_baseline/research_loop/`移至根目录`research_loop/`。
- `tools/`中的40个原有测试脚本移至根目录`tests/`。
- MLP辅助脚本按职责归类：`audit/`24个、`run/`34个、`verify/`7个、
  `reporting/`11个、`state/`4个、`data/`2个，共82个。
- MLP根目录保留模型、特征、训练和评估组件；`model.py`的内容与迁移前一致。

详细映射见[模块迁移表](../doc/module_path_migration.json)，使用方式见
[根目录说明](../README.md)和[测试说明](../tests/README.md)。

## 运行约定

测试从工程根目录执行：

```powershell
python -m pytest
```

训练、实验和打包从`paiec_baseline/`执行。举例：

```powershell
python -m mlp.data.make_split --help
python -m mlp.run.run_loop_round --help
```

报告和研究协议路径分别使用`../reports/`、`../research_loop/`；研究计划位于
`../doc/`。当前复现文档中的模块名已同步修改。

## 实际验证

- `python -m pytest -q`：110项测试通过。
- `python -m unittest discover -s tests -t .`：59项类测试通过；这些包含于上述pytest结果，不叠加计数。
- 核对323处内部MLP导入目标，全部存在。
- 核对当前入口及报告中的41个本地Markdown链接，全部可定位。
- 核对4个CLI的`--help`入口：训练、数据划分、实验驱动、CV汇总。
- 逐文件核对2025个历史记录、冻结协议、来源文件和`model.py`的SHA-256，均与迁移前一致。

本轮验证覆盖目录、导入、打包与已有算法测试；实验报告中的数值仍来自历史实验。
`runs/`中的源码快照、记录中的历史路径和冻结协议原文保留，按迁移表读取。

## 本地备份

迁移前工作文件及核验清单：
`D:\paiec_git_backups\layout_20261022_120427\`。
其中保存文件哈希清单、移动映射、测试输出及验证记录。
