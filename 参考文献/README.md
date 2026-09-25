# 参考文献

这 4 篇论文的 PDF 原文和 Markdown 全文转换稿只保存在本地这个文件夹里，不放进仓库：仓库是公开的，而 IEEE 论文有版权，arXiv 论文也不宜全文转载。需要原文时按下面的链接下载。

| 论文 | 出处 | 链接 | 在本方案中的用途 |
|---|---|---|---|
| H. Topcuoglu, S. Hariri, M.-Y. Wu. Performance-Effective and Low-Complexity Task Scheduling for Heterogeneous Computing | IEEE TPDS, 13(3): 260–274, 2002 | [DOI 10.1109/71.993206](https://doi.org/10.1109/71.993206) | HEFT / CPOP。调度器是不插入版的 HEFT；第 6 节的关键子任务前瞻（`child`）已实测，净效果约为零 |
| S. Kulagina, A. Benoit, H. Meyerhenke. Memory-aware Adaptive Scheduling of Scientific Workflows on Heterogeneous Architectures | CCGrid 2025 | [arXiv:2503.22365](https://arxiv.org/abs/2503.22365) | 主要借鉴来源：内存感知聚簇（`mc`）和 BLC 优先级（`blc`） |
| M. Wilhelm, T. Pionteck. Static task mapping for heterogeneous systems based on series-parallel decompositions | HCW 2025（IPDPS 研讨会） | [arXiv:2502.19745](https://arxiv.org/abs/2502.19745) | 串并联分解加模型驱动的局部搜索，尚未实现 |
| T. Zhu, D. Feng, E. Feng, Y. Xia. From Principles to Practice: A Systematic Study of LLM Serving on Multi-core NPUs | arXiv 预印本，2025 | [arXiv:2510.05632](https://arxiv.org/abs/2510.05632) | 只作引言中的硬件背景；文中“解析性能模型在访存密集场景下误差可达 38.56%”可用来支撑精确复刻 spill 的做法 |

每篇论文借鉴了什么、效果如何，见 `../00_操作日志_2026-09-24_文献改进.md` 第 2–5 节和设计文档 3.1、3.3 节。
