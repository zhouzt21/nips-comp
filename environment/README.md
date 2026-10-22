# PAIEC Windows CPU 环境迁移

本目录保存迁移包的构建脚本和精确依赖清单。生成的 `paiec-runtime-win64-py312-cpu.zip` 包含可直接启动的 **Windows x64 Python 3.12.7** 及 **44 个依赖**，其中 PyTorch 为 **2.9.1+cpu**，无需 GPU、CUDA、Anaconda 或目标机器预装 Python。

**按用户要求，不复制、不打包已经单独复制的 `D:\paiec_research_deps`。** 因而必须将该目录与本包配合使用。包不适用于 Linux、macOS 或 ARM；未在其他操作系统上验证。

## 1. 解压与指定外部依赖

将 ZIP 解压到目标机器，例如 `C:\paiec-env`，得到：

```text
C:\paiec-env\paiec-runtime-win64-py312-cpu\
  runtime\python.exe
  python.ps1
  verify_environment.py
  requirements-win-cpu.lock
  requirements-external.lock
  source_environment.json
  bundle_metadata.json
  file_manifest.json
  README.md
```

解压后约需 1 GB 空间（以实际清单为准），另为数据和缓存保留空间。建议使用路径较短的目录，避免旧 Windows 的长路径限制。

在 PowerShell 中指定**你已经复制的目录**，里面必须保留包目录及对应的 `.dist-info`：

```powershell
$env:PAIEC_RESEARCH_DEPS = 'D:\paiec_research_deps'
$paiecEnv = 'C:\paiec-env\paiec-runtime-win64-py312-cpu'
powershell -NoProfile -ExecutionPolicy Bypass -File "$paiecEnv\python.ps1" "$paiecEnv\verify_environment.py" --checksums
```

若外部依赖放到了其他盘，只需修改 `PAIEC_RESEARCH_DEPS`。也可给启动脚本传 `-DepsPath 'C:\your-copy\paiec_research_deps'`。`ExecutionPolicy Bypass` 只作用于当前子进程，不修改机器全局策略。

外部依赖的冻结版本如下，**无需重复安装或复制**：

| 包 | 版本 |
|---|---|
| huggingface-hub | 0.35.3 |
| transformers | 4.44.2 |
| tokenizers | 0.19.1 |
| safetensors | 0.4.5 |

这份外部依赖中有 Windows/Python 3.12 原生二进制，因此不能直接挪给其他系统或其他 Python 主版本。

包中已带应用本地的 Microsoft C/C++ 运行库 14.40，与原机正常工作的运行库一致，不自动安装系统组件。如目标 Windows 自检仍提示系统 DLL 缺失，可按微软的 [Visual C++ v14 x64 Redistributable 说明](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)检查系统环境。

## 2. 运行现有工程

项目源码请使用已有副本或从仓库克隆。下列测试需要目标机器安装 Git for Windows（个别打包测试会创建临时 Git 仓库）；普通训练、推理不依赖 Git。

```powershell
$project = 'E:\VScode_data\project\nips-comp' # 改成目标机器实际源码路径
Set-Location -LiteralPath $project
powershell -NoProfile -ExecutionPolicy Bypass -File "$paiecEnv\python.ps1" -m pytest -q

Set-Location -LiteralPath (Join-Path $project 'paiec_baseline')
powershell -NoProfile -ExecutionPolicy Bypass -File "$paiecEnv\python.ps1" -m mlp.train --help
```

也可以连同项目和本地语义模型做完整检查：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$paiecEnv\python.ps1" "$paiecEnv\verify_environment.py" --checksums --project "$project" --model-dir 'D:\paiec_pretrained\all-MiniLM-L6-v2' --output 'C:\paiec-env\verification.json'
```

请通过 `python.ps1` 启动；它会使用包内解释器、关闭用户 site-packages、清除原有 `PYTHONPATH`、隔离 Anaconda 的 PATH，并加载单独复制的外部依赖。只影响本次运行，结束后恢复原环境。线程数默认 OMP/MKL 为 4、OpenBLAS 为 1。控制台入口统一使用 `-m`，例如 `-m pip check`；包中不保留包含构建机绝对路径的 `pip.exe` 等启动器。

默认禁止 Hugging Face 联网加载，缓存放在迁移包的 `cache/huggingface` 内。确实需要重新下载数据时，在脚本参数前加 `-Online`，并在新环境重新登录自己的账号；本包不含任何令牌。例如：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$paiecEnv\python.ps1" -Online -m huggingface_hub.commands.huggingface_cli login
```

## 3. 环境以外仍需迁移的内容

本包只处理解释器与依赖。数据、研究缓存、预训练模型、训练权重和提交 ZIP 不属于 Python 环境，**没有自动放入本包**。

| 原路径 | 用途 | 本机审计体积，约 |
|---|---|---:|
| `D:\paiec_payload`（其中 `data`） | 真实公开训练数据 | 0.74 GiB |
| `D:\paiec_pretrained\all-MiniLM-L6-v2` | 离线语义编码模型 | 0.085 GiB |
| `D:\paiec_research_cache` | 语义向量及各轮 episode 缓存 | 1.92 GiB |
| 项目内未被 Git 跟踪的 `weights.npz`、`runs` 二进制、`dist` | 已训练模型和提交产物 | 依实际选择 |

`paiec_research_deps` 已由你复制；不要再从原机用户目录或整个 Anaconda 复制其他依赖。源码可以从 `https://github.com/zhouzt21/nips-comp.git` 获取，但 Git 仓库不包含上述本地数据与权重。

**现有研究脚本仍有 `D:/paiec_...` 固定资源路径。** 环境目录和外部依赖目录可改变位置，但要直接复现历史实验，建议在目标 Windows 上保留这些 D 盘资源路径，并从 `paiec_baseline` 工作目录运行。`mlp.train` 支持 `--data-root`；其他命令请按各自 `--help` 指定路径，不要假定所有实验脚本都支持同一个路径参数。

未将研究代码批量改写为新路径，因为部分语义缓存键取决于源码字节哈希，改写代码会改变缓存指纹。若目标设备没有 D 盘，环境自检和不需要数据的测试仍可执行；历史实验的资源路径与缓存指纹需要另行适配。迁移包不能将路径可用性、环境可用性与完整实验复现混为一谈。

MiniLM revision：`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`；公开数据 revision：`54074453da7b9911be0f48e4c560be547bc84d21`。数据和模型保留各自原始许可；本包中的 Python 和 wheel 许可也予以保留。没有复制 HF、Codabench、API 登录凭据。

## 4. 版本、验证与重建

依赖来自官方 PyPI 和 PyTorch CPU wheel index，Python 来自官方 NuGet `python 3.12.7` 包。安装后的依赖文件已打包，迁移运行不需要再下载 wheels。`bundle_metadata.json` 记录构建来源、wheel 哈希和源码版本，`file_manifest.json` 记录包内文件 SHA256；ZIP 旁的 `.sha256` 校验整个归档。

与原机的唯一 Python 依赖版本调整是 `kiwisolver 1.4.4 → 1.4.5`：原版本来自 Conda，没有 CPython 3.12 Windows wheel；该包用于 Matplotlib 绘图布局。另一个字符串差异是原机 torch 的发行元数据写作 `2.9.1`，实际运行版本已经是 `2.9.1+cpu`，本包使用匹配的官方 CPU wheel。构建产物使用 PyPI wheels，因此不保证与原 Conda 二进制构建逐字节相同。

实际验证还发现：PyArrow 16.1 wheel 内自带 `MSVCP140.dll 14.28`，先导入 PyArrow/pandas 再导入 Torch 2.9.1 会出现 `WinError 1114`。包内用原机同款 14.40 运行库替换该 DLL，并更新安装记录；来源为 Anaconda 的 `vs2015_runtime-14.40.33807-h98bb1dd_1`，11 个 DLL 与原机逐字节一致。没有通过更改研究代码导入顺序绕开问题。该兼容修复及其前后哈希记在 `bundle_metadata.json` 中；相关问题见 [PyTorch 官方问题记录](https://github.com/pytorch/pytorch/issues/166628)。

验证脚本检查版本、加载来源、CPU 反向传播、Parquet 读写、绘图、`pip check`，可选检查整个包的哈希、项目测试与离线 MiniLM 前向。目标机器仍需执行一次自检；在原机搬到不同目录运行通过，不代表已经在另一台物理设备上运行过。

如需重建，在 Windows x64 Python 3.12 环境中从项目根目录运行：

```powershell
python environment/build_windows_bundle.py --work 'D:\paiec_environment_rebuild'
```

该命令需要网络，只下载基础依赖和解释器，不下载、复制外部研究依赖。产物体积较大，`--work` 必须放在 Git 工程之外且空间充足的位置。打包完成后，按前述自检命令检查解压副本。

设计依据：[Python Windows 分发说明](https://docs.python.org/3.12/using/windows.html#the-nuget-org-packages)、[venv 的迁移限制](https://docs.python.org/3.12/library/venv.html)、[pip download 离线依赖准备](https://pip.pypa.io/en/stable/cli/pip_download/)。本包使用独立解释器和重新安装的依赖，未直接复制不能可靠搬动的原始 venv。
