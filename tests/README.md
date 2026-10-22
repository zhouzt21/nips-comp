# 测试入口

从`nips-comp/`根目录运行完整测试集：

```powershell
python -m pytest
```

`pytest.ini`限定从`tests/`收集测试。`tests/__init__.py`使测试能导入
`paiec_baseline/`中的现有预测器及工具，不需要复制模型或修改系统级Python路径。

只运行标准库unittest类测试（不包含pytest函数测试）：

```powershell
python -m unittest discover -s tests -t .
```

如果当前位于`paiec_baseline/`：

```powershell
python -m pytest ../tests
```

依赖包括pytest、NumPy、pandas、PyArrow、SciPy、PyTorch及现有工具的运行依赖。
测试使用合成样本或临时文件，并包含CPU训练/导出检查、协议隔离、ZIP打包、
源码发布检查及目录迁移检查。它们不重新运行公开数据实验，也不代表比赛分数。
