"""开发期快速核对：求解 + 官方评估，打印预测值与官方 makespan。

python -m solver.dev_check case_019 case_050 --cores 4 --problems 1 2
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from .config import SOLUTION_DIR, case_path, load_config
from .evaluate import run_official, single_core_makespan
from .graph import Graph
from .solve import solve

OUT = os.path.join(SOLUTION_DIR, 'results', 'dev')


def one(job):
    case, n, q = job
    G = Graph(case_path(case))
    plan, info = solve(G, n, q, load_config())
    path = os.path.join(OUT, '%s_p%d_n%d_plan.json' % (case, q, n))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(plan, f)
    r = run_official(case, q, path, OUT, tag='p%d_n%d' % (q, n))
    return case, n, q, info, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cases', nargs='+')
    ap.add_argument('--cores', type=int, nargs='+', default=[4])
    ap.add_argument('--problems', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        singles = dict(zip(args.cases, ex.map(single_core_makespan, args.cases, [OUT] * len(args.cases))))
        jobs = [(c, n, q) for c in args.cases for n in args.cores for q in args.problems]
        for case, n, q, info, r in ex.map(one, jobs):
            s = singles[case]
            pred = info['predicted'] or 0
            print('%s P%d N=%d  single=%d  official=%d  speedup=%.2f  predicted=%.0f (err %+.0f%%)  '
                  'subgraphs=%d  added=%dB spill=%dB hit=%s  params=%s  eval=%.0fs' % (
                      case, q, n, s, r['makespan'], s / r['makespan'], pred,
                      100.0 * (pred - r['makespan']) / r['makespan'], info['subgraphs'],
                      r['added_copy_bytes'], r['spill_added_copy_bytes'],
                      '%.2f' % r['cache_hit_rate'] if r['cache_hit_rate'] is not None else '-',
                      info['params'], r['eval_seconds']), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
