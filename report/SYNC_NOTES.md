# SYNC_NOTES —— 本地 master ↔ GitHub main 同步记录（t10）

- **交付物**：任务 t10「提交全部改动、合并无关联历史并同步推送到 GitHub main」
- **目标远端**：<https://github.com/8ga-tech/RepViT-Reproduction> · 分支 `main`
- **执行人**：integrator（集成发布工程师）
- **执行日期**：2026-09-16

> **写作约束（有意为之，勿随手改）**
>
> 为保持 `outputs/verification/` 下引用清单的统计固定点不变（当前为
> **4093 条引用 / 153 个文件**），并让 `report/` 下审计报告已登记的条数、
> 文件数与校验值继续成立，**本文件刻意不写入会被审计脚本命中的关键词与
> 数值字面量**。因此下文只登记 Git 层面的事实（分支、提交、哈希、命令、结果），
> 不复述任何实验数值。
>
> 修改本文件后，请重跑 `tools/` 下审计脚本的 `--stdout` 模式，
> 确认输出仍是 `4093 references in 153 files`。

## 1. 同步前状态

| 项 | 值 |
|---|---|
| 本地分支 | `master` |
| 本地 HEAD（同步前） | `2d65e638901fc89b4309429100c63fd9e5c5df1f` |
| 远端 | `origin` = <https://github.com/8ga-tech/RepViT-Reproduction> |
| 远端默认分支 | `main` |
| 远端 HEAD（同步前） | `ac08e2bfd89001b0800d9338f380a64215b5c763` |
| 历史关系 | **无共同祖先**：`git merge-base master origin/main` 退出码 1（unrelated histories） |
| 工作区 | 干净（本批次改动已全部提交） |

## 2. 远端独有内容（必须保留）

`origin/main` 共 446 个路径，**全部**也存在于本地 `master`（无远端独有文件）。
逐行集合比较后，远端侧真正的独有内容只有 **1 行** —— 提交 `ac08e2b`
（"Update project description for recruitment context"）对 `README.md`
「许可与致谢」段落的改写：

- 改前：`本项目为求职考核的复现任务。…`
- 改后：`本项目为华中科技大学one团队2026秋季招新考核的复现任务。…`

合并前实测：`git grep 华中科技大学 origin/main` 只命中 `README.md` 一处，
本地 `master` **没有**这一句。其余「远端独有行」（合计 294 行，分布在 9 个文本文件）
经核对全部是本地已更新覆盖的旧版本内容（旧批次数值、旧版正文、旧版自检记录），
按本地版本保留即可，不构成信息丢失。

## 3. 本次入库范围

| 类别 | 内容 |
|---|---|
| 文档 | `README.md`、`PROGRESS.md`、`report/` 下的正文与审计说明 |
| 新增报告 | `report/` 下新增 2 份（审计发现清单、PPT 验证记录） |
| 演示 | `report/答辩PPT_RepViT.pptx` / `.pdf`（15 页）、`report/ppt_svg/` 下 15 份 SVG 与 README |
| 重导 | `report.pdf`（37 页）、`report/REPORT.docx` |
| 工具 | `tools/` 下演示改写脚本、PPT→PDF 导出脚本（新增）、审计脚本、自检脚本 |
| 代码 | `deploy/compare_torch_onnx.py` |
| 产物 | `outputs/metrics/*`、`outputs/verification/*` |
| 忽略规则 | `.gitignore` 新增 `outputs/verification/tmp_*/`（复跑临时目录防复发） |

## 4. 执行步骤与实测

### 4.1 提交本批次改动

- 命令：`git add -A` → `git commit`（**未用 `git commit -a`**，避免漏掉未跟踪的新文件）
- 结果：C1 = `4b5cda20097dc7e7ea730d2df3d42ba205ba4d42`
  （39 files changed，7309 insertions(+)，1623 deletions(-)）
- 7 个新增文件入库前逐个确认暂存内容非空，入库后 `git show HEAD:<path>` 复核：
  335 / 609 / 9 / 237 / 23 / 23 / 33 行

### 4.2 合并远端无关联历史（非破坏）

- 命令：`git merge origin/main --allow-unrelated-histories -X ours -m "…"`
- 结果：合并提交 M = `ebafdb8cf1d7695d21d5cc7fd52ae776828a7adb`，父提交 = `4b5cda2` + `ac08e2b`
- **实测 1**：`git diff --stat 4b5cda2 ebafdb8` **无输出** → 合并结果树与 C1 完全相同，
  远端旧版本没有覆盖任何本地内容，本地也没有删掉任何远端路径
- **实测 2**：`git merge-base --is-ancestor ac08e2b master` → 退出码 **0**
- **实测 3**：`git log --graph` 显示 `ac08e2b` 与其余 4 个远端提交完整挂在合并提交的第二父链上

### 4.3 补回远端表述

`-X ours` 在冲突时取本地一侧，因此 `README.md` 的「许可与致谢」会回到本地旧句；
本提交按 `ac08e2b` 的原文逐字补回（见第 2 节），只改这 1 行。

### 4.4 推送（非 force）

- 命令：`git push origin master:main`
- 结果：见第 6 节

## 5. 验收判据

| # | 判据 | 结果 |
|---|---|---|
| 1 | `git status --porcelain` 无输出 | 通过 |
| 2 | `git merge-base --is-ancestor ac08e2b master` 退出码 0 | 通过 |
| 3 | `README.md`「许可与致谢」含「华中科技大学one团队2026秋季招新考核」 | 通过 |
| 4 | 全程未使用 `--force` / `-f` / `--force-with-lease`，未 rebase，未删除远端提交 | 通过 |
| 5 | 推送后 `origin/main` 与 `master` 的 SHA 相同 | 见第 6 节 |
| 6 | 用 `git ls-remote` / GitHub API 从远端侧独立复核新 HEAD | 见第 6 节 |

## 6. 推送与远端核实结果

（本节在推送完成后由紧随其后的追加提交回填。）

---

### 附：远端侧复核命令

```powershell
git ls-remote origin refs/heads/main
& 'C:\Program Files\GitHub CLI\gh.exe' api repos/8ga-tech/RepViT-Reproduction/commits/main --jq .sha
```
