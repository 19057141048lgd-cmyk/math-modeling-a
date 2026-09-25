# 实验记录：临时脚本与输出

2026-09-24 两轮改进中写在系统临时目录（`%TEMP%\mm_extract\`）里的诊断和实验脚本，以及它们的输出，整理到这里保存。

- **路径**：脚本里写死了 `C:\Users\richd\Desktop\数学建模\A题(1)\A题方案` 和 `%TEMP%\mm_extract\` 这类路径，换机器运行前要先改。
- **编码**：`输出/` 里的旧日志原来是编码错乱的 UTF-16，已转成 UTF-8；个别被截断的中文字符显示为替换符，不影响数字。

## 脚本（`脚本/`）

| 文件 | 用途 | 对应文档 |
|---|---|---|
| `diag_spill.py`、`diag_inputs.py`、`diag_groups.py` | 诊断 spill 最严重的用例：峰值驻留的组成、共享图输入、作业结构 | 操作日志第 3 节 |
| `exp_split0.py`、`exp_variants.py` | `mc`、`mr`、`blc` 等开关的单用例官方对比 | 操作日志第 3 节、5.4 节 |
| `compare_v2.py` | v1 与 v2 全量结果逐行对比 | 操作日志 5.2 节 |
| `recheck_review.py` | 复审报告 6 个问题在 v2 上的复核 | 操作日志第 9 节 |
| `diag_035.py`、`diag_035b.py`、`diag_035c.py`、`diag_035d.py` | case_035 问题2、5 核：官方逐算子时间线与模型对比，定位 MTE3 按序发射问题 | 设计文档 7.6 节 |
| `exp_cache.py` | 新旧 Cache 代理模型在 23 个验证用例、3 核和 5 核上的对比（v3） | 设计文档 7.6 节 |
| `test_cache.py`、`test_rows_cache.py` | v3 的 Cache 模型单元检查、续跑缓存校验检查 | 设计文档 7.6 节 |
| `eval_cost.py` | 按图规模统计官方评估耗时，用来定 `--verify-small` 的阈值 6000 | 设计文档 7.6 节 |
| `stamp_single.py` | 给已有的 100 个单核基准补写校验文件（`results/single/*.key`），没有重算 | 设计文档 7.6 节 |
| `a_full.py`、`a_full2.py`、`v2_extra.py` | 从 `results_all/summary.csv` 统计设计文档 7.4 节的全部数字，只读结果，不调用评估器 | 设计文档 7.4 节 |

## 输出（`输出/`）

| 文件 | 内容 |
|---|---|
| `v2_full.txt`、`v2_full2.txt`、`v2_extra.txt` | 上面三个统计脚本在 v2 `results_all/` 上的输出 |
| `exp_split0_b.txt`、`exp_var_b.txt` | 开关单用例对比的输出 |
| `cache_exp/`、`cache_exp_log.txt`、`cache_exp_log2.txt` | Cache 模型对比实验：计划 46 组，完成 42 组，还没有汇总；失败的几组是内存耗尽造成的 |
| `diag035/` | case_035 诊断用的方案、官方结果和逐算子 trace |
