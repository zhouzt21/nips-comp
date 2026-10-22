# 运行环境迁移记录（2026-10-22）

## 范围

为 Windows x64 生成独立 CPU 运行环境，工作目录为 `D:\paiec_migration_20261022`。Python 3.12.7，PyTorch 2.9.1+cpu；44 个包内发行依赖，加上用户已复制的 4 个外部依赖。

按用户要求，**不重新复制 `D:\paiec_research_deps`**。其中 `huggingface-hub 0.35.3`、`transformers 4.44.2`、`tokenizers 0.19.1`、`safetensors 0.4.5` 由启动脚本通过可配置路径加载。原机 Anaconda、用户 site-packages、Prophesee PYTHONPATH 不参与迁移包运行。

源码基点为 `9618657ae3486110c41966ae34c87ad7a4fd9711`；本次新增 `environment/` 中的锁文件、原环境审计、构建脚本、启动器、自检与操作说明，并在根 README 添加入口。未修改研究算法、实验记录、权重或数据。

## 发现与修复

1. 原环境混用了 Anaconda、用户 site-packages、D 盘额外依赖，因此重新组装独立解释器与精确版本依赖。
2. 原 Conda 的 `kiwisolver 1.4.4` 缺少 cp312 Windows wheel，迁移用 `1.4.5`，并实际检查 Matplotlib PNG 输出。
3. PyArrow 16.1 wheel 中的 `MSVCP140.dll 14.28` 导致先导入 pandas/PyArrow、再导入 Torch 时发生 `WinError 1114`。分别导入 NumPy、SciPy、sklearn 后再导入 Torch 未复现；单独导入 Torch 也正常。
4. 从原机同款 `vs2015_runtime-14.40.33807-h98bb1dd_1` 包恢复 11 个 14.40 DLL，与原环境逐字节一致；同时替换 PyArrow 内的旧 DLL并更新 RECORD。此前失败的完整导入顺序随后通过。只放新 DLL 在解释器根目录不足以解决问题，该失败尝试也保留在本机诊断记录中。
5. 自检发现 PowerShell 位置参数会误绑定外部依赖路径，已明确参数位置；脚本参数及 `-m mlp.train --help` 均实际执行通过。

## 已执行的验证

在包内独立解释器、禁用用户 site-packages、净化 PATH 的条件下：

- 48 个依赖的版本与来源校验通过，其中 4 个来自外部依赖目录。
- NumPy、pandas、PyArrow、SciPy、sklearn、Torch、Transformers 等导入通过。
- CPU 前向、反向与 AdamW 更新通过。
- Parquet 写入/读回、Matplotlib 绘图通过。
- `pip check`：`No broken requirements found.`
- 本地 MiniLM 离线加载及前向通过，输出维度 384。
- 项目全套测试：**110 passed in 10.63s**。

原始结果位于 `D:\paiec_migration_20261022\verification-build.json`。

最终 ZIP 已解压到不同目录 `D:\paiec_migration_check\relocated runtime`（路径含空格），从该副本启动并完成：

- **24,295 个清单文件的大小与 SHA256 全部匹配**。
- 48 个依赖的版本、加载来源及全部原生库检查通过。
- CPU 训练步骤、Parquet、绘图、`pip check`、离线 MiniLM 再次通过。
- 项目全套测试再次通过：**110 passed in 22.47s**。

最终证据：`D:\paiec_migration_20261022\verification-relocated.json`；归档审计：`D:\paiec_migration_20261022\archive_verification.json`。验证在本机不同路径完成，未声称已在另一台物理设备运行。

## 最终交付

迁移包：`D:\paiec_migration_20261022\paiec-runtime-win64-py312-cpu.zip`。

- 归档大小：265,807,501 bytes，约 **253.49 MiB**。
- 解压后文件合计约 **855.01 MiB**。
- ZIP 共 24,296 个条目，无重复路径；未包含四个外部依赖的模块文件。
- 外部 SHA256 文件：同路径后缀 `.zip.sha256`。

```text
cd79e4c2f8803db529ef0ca6687a23638f56f29b02cd772fcd5e285b4b74edef
```

ZIP 内说明与项目 `environment/README.md` 内容一致。构建日志、依赖下载日志和完整包内校验清单保存在 D 盘构建目录；大体积归档未加入 Git。

## 使用与证据边界

操作说明见 [环境迁移 README](../environment/README.md)。包中包括解释器、已安装依赖、启动脚本、版本锁和 SHA256 清单；没有数据、MiniLM 权重、研究缓存、训练产物或访问令牌。外部依赖由用户之前的副本提供。

环境目录和启动器使用的外部依赖路径可改变；部分历史实验仍固定使用 `D:/paiec_...`，资源迁移需保持相应路径，或按实际 CLI 调整。其中 `mlp/run/run_pool_scale_round.py` 会自行覆盖子进程的 `PYTHONPATH` 为原 D 盘路径，运行该历史驱动时也需保持此路径或单独适配。未批量改写这些源码，避免改变源码绑定的缓存指纹。

迁移测试验证软件运行与接口可用性；不等同于重新完成全部真实数据实验，也不增加模型跨 benchmark 泛化证据。Linux、macOS、ARM 不在此包支持范围内。
