"""对照组：朴素基线（独立作业整体 LPT 分配、每核一个 Task）在问题 1/2 评估器下的加速比。

python -m solver.run_baseline --cases case_019 case_050 --cores 2 3 4 5
结果写入 results/baseline.csv
"""
import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from .config import SOLUTION_DIR, case_path, load_config
from .evaluate import run_official, single_core_makespan
from .graph import Graph
from .solve import build_candidate

RESULTS = os.path.join(SOLUTION_DIR, 'results')


def one(job):
    case, n = job
    G = Graph(case_path(case))
    plan, _, n_sub, _, _ = build_candidate(G, n, load_config(), 'A', math.inf, 16, 0.0)
    out = os.path.join(RESULTS, 'baseline')
    path = os.path.join(out, '%s_n%d_plan.json' % (case, n))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(plan, f)
    return case, n, [run_official(case, q, path, out, tag='base_p%d_n%d' % (q, n))['makespan'] for q in (1, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', nargs='+', required=True)
    ap.add_argument('--cores', type=int, nargs='+', default=[2, 3, 4, 5])
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(os.path.join(RESULTS, 'baseline'), exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        singles = dict(zip(args.cases, ex.map(single_core_makespan, args.cases,
                                              [os.path.join(RESULTS, 'single')] * len(args.cases))))
        rows = list(ex.map(one, [(c, n) for c in args.cases for n in args.cores]))
    with open(os.path.join(RESULTS, 'baseline.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['case', 'cores', 'single_makespan', 'p1_makespan', 'p1_speedup', 'p2_makespan', 'p2_speedup'])
        for case, n, (m1, m2) in rows:
            s = singles[case]
            w.writerow([case, n, s, m1, round(s / m1, 4), m2, round(s / m2, 4)])
    for n in args.cores:
        sp = [singles[c] / m[0] for c, k, m in rows if k == n]
        print('baseline N=%d  平均加速比(问题1评估)=%.3f' % (n, sum(sp) / len(sp)))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
