"""开发工具：标定 spill 字节到 makespan 的折算关系。

对给定用例的全部候选方案估计逐核 spill，把有 spill 的方案（按内容去重）交官方评估，
比较几种折算方式（B 为 DDR 带宽）：
  M1 = 预测 + a * max_k spill_k / B
  M2 = max_k (核 k 预测结束 + a * spill_k / B)
  M3 = 预测 + a * sum_k spill_k / B
  M4 = 预测 + a * u * sum_k spill_k / B，u = min(1, (模型 DDR 字节 + spill) / (B * 预测)) 为 DDR 利用率
用法（在 A题方案 目录下）：
    python -m solver.dev_spill_time [--workers 8]          # 16 个代表用例，写 records.csv
    python -m solver.dev_spill_time --cases case_014 case_073 --cores 4 --tag heldout --timeout 900
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from .config import SOLUTION_DIR, case_path, load_config
from .evaluate import run_official
from .graph import Graph, build_clusters
from .memory import estimate_spill
from .plan import plan_scene_a, plan_scene_b, validate_plan
from .sched import schedule
from .solve import CANDIDATES, plan_signature

OUT = os.path.join(SOLUTION_DIR, 'results', 'dev', 'spilltime')
CASES = ['case_001', 'case_004', 'case_010', 'case_019', 'case_026', 'case_029', 'case_044', 'case_046',
         'case_048', 'case_050', 'case_057', 'case_061', 'case_064', 'case_071', 'case_080', 'case_094']


def candidates_of(case, problems, cores, max_task_ops=None):
    G = Graph(case_path(case))
    hw = load_config()
    recs = []
    for q in problems:
        for n in cores:
            seen = set()
            for i, params in enumerate(CANDIDATES[q]):
                scene, split, grain, loc = params[:4]
                task_div = params[4] if len(params) > 4 else None
                clusters, _ = build_clusters(G, n, split, grain)
                cap = G.total_cycles / (n * task_div) if task_div else None
                S = schedule(G, clusters, n, scene, hw, loc, cap)
                plan = plan_scene_a(clusters, S) if scene == 'A' else \
                    plan_scene_b(clusters, S, G.total_cycles / (n * 40.0))
                validate_plan(G, plan)
                sig = plan_signature(plan)
                if sig in seen:
                    continue
                seen.add(sig)
                if scene == 'A' and max_task_ops and \
                        max(Counter(plan['node_to_subgraph'].values()).values()) > max_task_ops:
                    continue
                total, per_core, ok = estimate_spill(G, plan, scene, hw)
                if total == 0:
                    continue
                path = os.path.join(OUT, 'plans', '%s_p%d_n%d_%d.json' % (case, q, n, i))
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(plan, f)
                recs.append({'case': case, 'problem': q, 'cores': n, 'params': str(params), 'plan': path,
                             'pred': S.makespan, 'core_end': json.dumps(S.core_end),
                             'spill_core': json.dumps([per_core.get(k, 0) for k in range(n)]),
                             'spill': total, 'ddr_bytes': S.ddr_bytes, 'feasible': ok})
    return recs


def evaluate(rec, timeout):
    tag = os.path.splitext(os.path.basename(rec['plan']))[0]
    try:
        r = run_official(rec['case'], rec['problem'], rec['plan'], os.path.join(OUT, 'eval'), tag=tag,
                         timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    return dict(rec, official=r['makespan'], official_spill=r['spill_added_copy_bytes'],
                eval_seconds=round(r['eval_seconds'], 1))


def models(r, a, bw):
    end = json.loads(r['core_end'])
    sp = json.loads(r['spill_core'])
    pred, total = float(r['pred']), sum(sp)
    u = min(1.0, (float(r['ddr_bytes']) + total) / (bw * pred))
    return (pred + a * max(sp) / bw,
            max(e + a * s / bw for e, s in zip(end, sp)),
            pred + a * total / bw,
            pred + a * u * total / bw)


def fit(recs, bw):
    base = [abs(math.log(r['official'] / float(r['pred']))) for r in recs]
    print('有 spill 的方案 %d 个；不修正时 |log(官方/预测)| 均值 %.3f' % (len(recs), sum(base) / len(base)))
    for m, name in enumerate(('M1 max-core', 'M2 per-core', 'M3 sum', 'M4 ddr-util')):
        best = None
        for a10 in range(1, 61):
            a = a10 / 10
            err = [abs(math.log(r['official'] / models(r, a, bw)[m])) for r in recs]
            e = sum(err) / len(err)
            if best is None or e < best[0]:
                best = (e, a, max(err))
        print('  %-12s 最优 a=%.1f  |log| 均值 %.3f  最大 %.3f' % (name, best[1], best[0], best[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', nargs='+', default=CASES)
    ap.add_argument('--problems', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--cores', type=int, nargs='+', default=[2, 3, 4, 5])
    ap.add_argument('--tag', default='', help='结果写入 records_<tag>.csv（默认 records.csv）')
    ap.add_argument('--timeout', type=float, default=None, help='单次官方评估的超时秒数，超时的方案跳过')
    ap.add_argument('--max-task-ops', type=int, default=None,
                    help='跳过最大子图超过该算子数的场景 A 方案（问题1 评估器耗时随 Task 规模近似平方增长）')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    for sub in ('plans', 'eval'):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    k = len(args.cases)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        recs = [r for lst in ex.map(candidates_of, args.cases, [args.problems] * k, [args.cores] * k,
                                    [args.max_task_ops] * k) for r in lst]
        print('候选中有 spill 的不同方案 %d 个，开始官方评估' % len(recs), flush=True)
        done = list(ex.map(evaluate, recs, [args.timeout] * len(recs)))
    skipped = sum(r is None for r in done)
    recs = [r for r in done if r is not None]
    name = 'records_%s.csv' % args.tag if args.tag else 'records.csv'
    with open(os.path.join(OUT, name), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    bad = [r for r in recs if r['official_spill'] != r['spill']]
    print('评估超时跳过 %d 个；spill 字节与官方不一致的方案 %d 个' % (skipped, len(bad)))
    fit(recs, load_config().bandwidth)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
