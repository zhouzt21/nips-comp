# 两套固定 ridge=4 的最终候选：复现入口

最终状态和分数见 `ACCEPTANCE.md`；机器可读记录见 `acceptance_summary.json`、`release_scores.json`、`build_manifest.json`。`hosted_score` 只有真实 Hosted 运行后才能填写。

## 产物

- `../dist/paiec_r25_nopeer_final_candidate.zip`
- `../dist/paiec_partial8_shared_final_candidate.zip`
- 独立构建目录：`packages/paiec_r25_nopeer_final_candidate/`、`packages/paiec_partial8_shared_final_candidate/`
- 每个 ZIP 根目录有 `model.py`、`requirements.txt`、依赖锁、来源清单和状态；各自内置 MiniLM，彼此无目录依赖。
- `artifacts/evaluation/`：完整历史回放和固定4新概率；`artifacts/adapter_evaluation/`：回退的完整计分核算。
- `fixtures/`：36 Benchmark × 4 query × 六预算原始公开输入及冻结缓存参考；不含 query outcomes，不进入 ZIP/Git。
- `logs/`、`artifacts/commands/`：实际命令与 stdout/stderr。`logs/attempt1/` 保留第一次 Windows mmap 清理失败和被中断的验收。

## 构建与评价（PowerShell，仓库 paiec_baseline 目录）

```powershell
$researchPython = 'D:\nips-comp\paiec-runtime-win64-py312-cpu\python.ps1'
& $researchPython -B final_candidates/refresh_official.py
& $researchPython -B final_candidates/evaluate_release.py
& $researchPython -B final_candidates/evaluate_adapter.py
& $researchPython -B final_candidates/make_fixtures.py
& $researchPython -B final_candidates/build.py
& $researchPython -B final_candidates/verify_source.py
```

评价需要原冻结输入、只读 embedding cache 和数据集；它们不是 ZIP 推理依赖。原数据 revision `54074453da7b9911be0f48e4c560be547bc84d21`，模型 revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。历史 `4/4/1` 与发布 `4/4/4` 不能混记。主 cohort 固定7501，种子0/1/2，预算0/1/3/7/15/31，未按新结果重新调参。

## 干净环境（Windows Python 3.12；设置阶段需要下载依赖）

```powershell
& $researchPython -m venv D:\nips-comp\paiec-final-acceptance\win312
$cleanPython = 'D:\nips-comp\paiec-final-acceptance\win312\Scripts\python.exe'
& $cleanPython -I -m pip --isolated install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
& $cleanPython -I -m pip --isolated install -r final_candidates/requirements.txt --index-url https://pypi.org/simple
# 完整复现已实测的传递依赖版本：
& $cleanPython -I -m pip --isolated install -r final_candidates/dependencies.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
& $researchPython -B final_candidates/accept.py
```

验收从实际 ZIP 解压到仓库外；使用新 venv、空 HF cache、`-I -B`、离线 HF 设置、网络审计和禁止外部研究数据路径的审计。推理阶段不需要下载模型或读取历史缓存。Python依赖需要预装/安装，ZIP不包含 Python 解释器与全部 wheel；未知 Hosted requirements 安装政策仍是就绪边界。

`accept.py` 调用当前本地和最新公开官方 ZIP 检查器，执行原始输入、六预算、0/1/2/3 Peer、身份歧义、重复响应、独立进程、资源及无状态测试。`public_streaming_probe.py` 对实际候选调用官方公开 streaming client，协调器交换仅为本地注入；不能当真实 Hosted。每个实际命令另存于 `artifacts/commands/`，包含路径和返回码。

## 发布范围

没有正式提交 Codabench，没有重训，没有修改 R1–R81 或 smoothed_mean_strength2.zip。Git 保存源码、精简指标、合同和证据索引；权重、ZIP、逐样本数据和大型记录留本地。`publication_scope.json` 记录本轮未纳入 Git 的文件路径与原因；历史保护记录见 `historical_preservation.json`。克隆仓库本身不能恢复被忽略的大型本地实验资产。
