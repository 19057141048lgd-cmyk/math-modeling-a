"""作业级分配的纯计算量分析（设计文档 1.4 节的复算脚本）。

python -m solver.dev_jobbound [--out results_all]

把图按中间张量的弱连通分量拆成独立作业，作业计算量 w_j = 作业内算子 cycles 之和，W = Σ w_j。
只做作业级分配（作业不拆开）时，N 核上任何分配的最大负载都不小于 max(W/N, max_j w_j)，
因此纯计算量口径下作业级加速比的上界为 min(N, W / max_j w_j)；LPT 分配的实际值 W / LPT 最大负载不超过该上界。
这是近似分析，不是官方口径的严格上界：官方单核基准含 M/V 流水线并行（小于 W）与 spill（可大于 W），
多核时还有搬运与屏障，所以个别用例的官方加速比可以超过它（如 case_050）。
若给出 --out，同时读取 <out>/summary.csv 与 <out>/baseline.csv，列出官方结果超过该近似界的用例数。
"""
import argparse
import csv
import glob
import heapq
import os
import sys

from .config import case_path, data_dir
from .graph import Graph


def lpt(ws, n):
    loads = [0.0] * n
    for w in sorted(ws, reverse=True):
        heapq.heapreplace(loads, loads[0] + w)
    return max(loads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cores', type=int, nargs='+', default=[2, 3, 4, 5])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    cases = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(data_dir(), 'case_*.json')))
    bound, by_lpt = {}, {}
    for case in cases:
        G = Graph(case_path(case))
        ws = [sum(G.cycles[u] for u in job) for job in G.jobs()]
        W = sum(ws)
        for n in args.cores:
            bound[(case, n)] = min(n, W / max(ws))
            by_lpt[(case, n)] = W / lpt(ws, n)
    print('| 核数 | 作业级上界 min(N, W/max w) 平均 | LPT 分配加速比平均 | 上界 < 1.5 的用例数 |')
    print('|---|---|---|---|')
    for n in args.cores:
        b = [bound[(c, n)] for c in cases]
        l_ = [by_lpt[(c, n)] for c in cases]
        print('| %d | %.4f | %.4f | %d |' % (n, sum(b) / len(b), sum(l_) / len(l_), sum(x < 1.5 for x in b)))
    big = sum(1 for c in cases if bound[(c, 2)] < 2)
    print('\n最大作业超过总计算量一半的用例数：%d' % big)
    if args.out:
        for name, cols in (('summary.csv', None), ('baseline.csv', ('p1_speedup', 'p2_speedup', 'p3_speedup'))):
            path = os.path.join(args.out, name)
            if not os.path.exists(path):
                continue
            rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
            for n in args.cores:
                if cols is None:
                    over = [(r['case'], r['problem']) for r in rows if int(r['cores']) == n
                            and float(r['speedup']) > bound[(r['case'], n)] + 1e-9]
                else:
                    over = [(r['case'], col) for r in rows if int(r['cores']) == n for col in cols
                            if float(r[col]) > bound[(r['case'], n)] + 1e-9]
                print('%s %d 核：官方加速比超过该近似界的组合 %d 个%s' % (
                    name, n, len(over), '，例如 ' + ', '.join('%s/%s' % o for o in over[:4]) if over else ''))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
