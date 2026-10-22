# 版本管理

## 已采用的结构

只有根目录仓库负责版本管理。`paiec_baseline/` 为普通目录。远程目标为：

`https://github.com/zhouzt21/nips-comp.git`

历史中的第一次提交保存完整、未修改的上游代码，标签为`paiec-original-cc88a95`；第二次提交保存迁移时已经存在的研究修改、根目录计划文档与可复现实验记录。后续在`main`继续正常提交。原版标签保持固定，不移动、不覆盖。

## 日常提交

从本工程根目录执行，按本次实际变更选择路径：

```powershell
git status --short
git add paiec_baseline/mlp paiec_baseline/tools paiec_baseline/runs tests reports research_loop doc
git diff --cached --stat
python paiec_baseline/tools/check_public_source.py --research-records
git commit -m "Describe the completed research change"
git push
```

研究计划位于`doc/`，提交前同样需要检查其变更。数据、缓存、权重和大型逐样本CSV仍被忽略；`git add`不会把它们上传。新增实验时，保留唯一实验ID、配置、数据revision、切分、随机种子、真实指标和失败记录。

`--research-records`显式允许`runs/`中的指定文本记录，仍检查凭据及禁止的产物路径。不加此选项时，检查工具保留上游拒绝运行记录的默认行为。`.gitattributes`为研究源码和证据保留原始字节，避免换行转换改变已有SHA-256记录；其他源码使用LF换行。

历史检索文件`research_loop/literature/raw/targeted_search.json`实际为Markdown检索文本。目录调整保留了原文；检查器在上述显式选项下识别原始检索目录中以`## Query`开头的文本，仍扫描凭据。实验配置JSON继续要求能够正常解析。

## 查看原版和比较改动

```powershell
git show --stat paiec-original-cc88a95
git diff paiec-original-cc88a95 HEAD -- paiec_baseline
git log --oneline --decorate
```

需要在另一个目录查看原始文件时，选择一个尚不存在的目录，例如：

```powershell
git worktree add --detach 'D:\paiec-original-view' paiec-original-cc88a95
```

这样可以查看原版，当前开发目录继续保留修改版。原版快照仅包含上游文件，所以没有后来新增的`mlp/`及本项目根目录文档。

## 迁移备份

本次完整本地备份：`D:\paiec_git_backups\nips-comp_20261022_113534\`

- `workspace/`：迁移前全部工作文件和内外层Git元数据；2544个文件逐文件SHA-256校验通过。
- `original_child_history.bundle`：原子仓库历史，已通过`git bundle verify`。
- `backup_manifest.json`：迁移前文件清单与哈希。
- `migration_journal.json`：本次迁移的提交、标签及核验记录。
- `detached_child_git/`：从工作区转移出的原子仓库Git元数据。

Windows跨盘移动隐藏目录时遗留的空目录已重命名为`paiec_baseline/.git-detached-empty`，没有Git语义且被忽略。它不含源码或历史。备份保留在本机，没有加入GitHub仓库。
