# 执行合同澄清与审查留痕

2026-10-22 18:28 +08。R54已开始，原冻结合同与核心源文件hash保留。

1. 原合同“原patience5”仅是普通MLP的历史规则。历史R43 `reliability_training.py` 固定20epochs，`run/run_reliability_round.py` 对两个来源方向的合并曲线取含epoch0的argmin，没有patience。R54给R43增加的patience5是**反事实选择对照**，不是历史实际早停；不能声称R43曾因该规则被截断。所有数值配置与执行不变。
2. 1800秒限制当前在epoch开始检查。可能允许最后一个epoch越过限额；非有限loss会直接抛异常而非自动写incomplete.json。实际fit耗时与完成记录将逐项核查，任何超限/失败fit不得被归入完整负结果。后续入口加异常状态记录；不为此改变已开始fit的优化器或数据实现。

来源：单个只读structural reviewer针对复杂复现合同的核对。不是双审，也不自动代表科研claim通过。
