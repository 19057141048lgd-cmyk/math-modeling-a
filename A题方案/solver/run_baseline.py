"""对照组：朴素基线（独立作业整体 LPT 分配、每核一个 Task）在问题 1/2/3 评估器下的加速比。

python -m solver.run_baseline --cases case_019 case_050 --cores 2 3 4 5
python -m solver.run_baseline --cases all --workers 8 --out results_all     # 全部 100 个用例
结果写入 <out>/baseline.csv，方案与官方输出在 <out>/baseline/；已有的官方输出直接复用，可中断后续跑。
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from .config import SOLUTION_DIR, case_path, data_dir, load_config
from .evaluate import read_result, run_official, single_core_makespan
from .graph import Graph
from .solve import build_candidate

RESULTS = os.path.join(SOLUTION_DIR, 'results')


def one(job):
    case, n, out = job
    path = os.path.join(out, '%s_n%d_plan.json' % (case, n))
    if not os.path.exists(path):
        G = Graph(case_path(case))
        plan, _, _, _, _ = build_candidate(G, n, load_config(), 'A', math.inf, 16, 0.0)
        with open(path + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(plan, f)
        os.replace(path + '.tmp', path)
    ms = []
    for q in (1, 2, 3):
        res = os.path.join(out, '%s_base_p%d_n%d.json' % (case, q, n))
        ms.append(read_result(res)['makespan'] if os.path.exists(res) else
                  run_official(case, q, path, out, tag='base_p%d_n%d' % (q, n))['makespan'])
    return case, n, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', nargs='+', required=True, help='用例名列表，或 all')
    ap.add_argument('--cores', type=int, nargs='+', default=[2, 3, 4, 5])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default=RESULTS, help='结果目录（默认 results/）')
    args = ap.parse_args()
    cases = args.cases
    if cases == ['all']:
        cases = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(data_dir(), 'case_*.json')))
    cases = sorted(cases, key=lambda c: -os.path.getsize(case_path(c)))
    out = os.path.abspath(args.out)
    sub = os.path.join(out, 'baseline')
    os.makedirs(sub, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        singles = dict(zip(cases, ex.map(single_core_makespan, cases,
                                         [os.path.join(RESULTS, 'single')] * len(cases))))
        rows = list(ex.map(one, [(c, n, sub) for c in cases for n in args.cores]))
    rows.sort(key=lambda r: (r[0], r[1]))
    with open(os.path.join(out, 'baseline.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['case', 'cores', 'single_makespan', 'p1_makespan', 'p1_speedup', 'p2_makespan', 'p2_speedup',
                    'p3_makespan', 'p3_speedup'])
        for case, n, (m1, m2, m3) in rows:
            s = singles[case]
            w.writerow([case, n, s, m1, round(s / m1, 4), m2, round(s / m2, 4), m3, round(s / m3, 4)])
    for n in args.cores:
        sp = [[singles[c] / m[q] for c, k, m in rows if k == n] for q in range(3)]
        print('baseline N=%d  平均加速比 问题1=%.3f 问题2=%.3f 问题3=%.3f' % (
            n, *(sum(v) / len(v) for v in sp)))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
