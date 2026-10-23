# R51/R52 实现和合同核对

2026-10-22。该记录区分实现核对与方法效果；通过实现检查不代表通过 Research Gate。

## 数值与接口检查

迁移运行环境下合并执行四个测试模块，**28 tests passed**，exit code 0；unittest计时4.201秒。模块为 `test_neural_process.py`（7）、`test_amortized_irt.py`（8）、`test_reproduction_pipeline.py`（6）、`test_irt_training.py`（7）。原输出：`paiec_baseline/runs/research_loop/reproduction_v4/implementation_tests.log`。

覆盖作者核心方程/NumPy对照、expected-NLL loss/gradient、150 nuisance共享语义、分块目标等价、重复response计数、support/query隔离、fit-only变换、pair/BM权重、六预算、MAP驻点与置换不变、ridge截距、消融命名、checkpoint预测一致、冷启动全流程及timeout=incomplete。合成微型拟合不是实际性能实验。

实际数据检查：三个来源池R25回放误差为−4.698×10⁻¹²、−4.028×10⁻¹²、−1.037×10⁻¹²，见 `paiec_baseline/runs/research_loop/reproduction_v4/source_fold*.json`。预算、采样和强基线沿用原值。

## 独立只读结构复核

依据用户全局AGENTS中“复杂复现合同可使用单一structural_reviewer”，使用一个独立只读agent核对R51及R52的数据/方法/门槛/恢复合同。未使用Quality Judge，未声称双审。复核本身未运行重训练。

R51最初指出依赖provenance不完整、末epoch超时恢复可能误记完成；均在真实实验前修复并复核，无剩余阻断项。

R52复核无阻断项：expected-NLL与作者固定源码一致；fit-only校准/θ/ridge；pair等权计数；独立zerohead；三pool AND；固定收敛及资源上限；原文、依赖、训练输入与checkpoint哈希恢复。复核版本如下：

| 文件 | SHA256 |
|---|---|
| `R52_protocol.md` | `c6458659b2a28caeef173b546546f1e46b5eac5526daefea8861b408612b49ea` |
| `mlp/irt_training.py` | `4e2b0f2dfe663ce56c43160ad35feb0d33d78298d71e9b164def0960c539ec06` |
| `mlp/run/run_amortized_irt_round.py` | `21963fd77475161588f54d6a75525102e2be8306b24e81075d9676e5e6e9e708` |
| `tests/test_irt_training.py` | `f16dd4a2a652347a1cd5bdda50b539d3f48a4731e1d667ce6417f2c5ac1f4f8d` |

来源静态对齐不等于运行原作者全部实验；论文与代码不一致时以记录的具体方法边界表述。R51图已本地查看，标题/来源池/early-stop点/基线可读；该检查仅涉及绘图表达。

## 实验完成后的工件核对

R51全部12个history均为epoch0–60；R52全部6个fit summary为complete/converged且梯度≤1e-5。两轮source gate全部false。R51的28个和R52的53个provenance依赖逐一重新计算SHA256，均与评分前一致；用户原始prompt也保持一致。198行六预算指标表已生成。检查记录为 `paiec_baseline/runs/research_loop/reproduction_v4/final_artifact_checks.json`。

R52六预算图已本地查看；恒定0.5、R25、primary与zero-head曲线及来源池标签清晰。环境与750个本机权重/nuisance文件hash在 `execution_inventory.json`，不是已上传模型。两个训练进程均以exit code 0结束，没有后台训练待完成。
