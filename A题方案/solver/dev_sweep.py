"""开发期参数扫描：同一用例在不同 (split, grain, locality) 下的模型预测 vs 官方结果。"""
import itertools
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from .config import SOLUTION_DIR, case_path, load_config
from .evaluate import run_official, single_core_makespan
from .graph import Graph, build_clusters
from .plan import plan_scene_a, plan_scene_b, validate_plan
from .sched import schedule

OUT = os.path.join(SOLUTION_DIR, 'results', 'dev', 'sweep')
SCENE = {1: 'A', 2: 'B', 3: 'C'}


def one(job):
    case, n, q, split, grain, loc = job
    G = Graph(case_path(case))
    clusters, _ = build_clusters(G, n, split, grain)
    S = schedule(G, clusters, n, SCENE[q], load_config(), loc)
    if q == 1:
        plan = plan_scene_a(clusters, S)
    else:
        plan = plan_scene_b(clusters, S, G.total_cycles / (n * 40.0))
    nsg = validate_plan(G, plan)
    tag = 'p%d_n%d_%s_%d_%s' % (q, n, split, grain, loc)
    path = os.path.join(OUT, '%s_%s_plan.json' % (case, tag))
    json.dump(plan, open(path, 'w', encoding='utf-8'))
    r = run_official(case, q, path, OUT, tag=tag)
    return job, S.makespan, r['makespan'], nsg, r['added_copy_bytes']


SPLITS = [math.inf, 1.0, 0.25, 0.1]
GRAINS = [16, 64]
LOCS = [0.0, 1.0, 2.0, 4.0, 8.0]

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    os.makedirs(OUT, exist_ok=True)
    n, qs, cases = int(sys.argv[1]), [int(x) for x in sys.argv[2].split(',')], sys.argv[3:]
    with ProcessPoolExecutor(max_workers=8) as ex:
        singles = dict(zip(cases, ex.map(single_core_makespan, cases,
                                         [os.path.join(SOLUTION_DIR, 'results', 'dev')] * len(cases))))
        grid = list(itertools.product(cases, [n], qs, SPLITS, GRAINS, LOCS))
        rows = list(ex.map(one, grid))
    table = {}
    for job, pred, real, nsg, added in rows:
        table.setdefault((job[0], job[2]), []).append((job[3:], pred, real))
    wins = {}
    for (case, q), items in sorted(table.items()):
        s = singles[case]
        off = min(items, key=lambda r: r[2])
        mod = min(items, key=lambda r: r[1])
        base = [r for r in items if r[0][0] == math.inf and r[0][2] == 0.0][0]
        print('%s P%d N=%d  baseline=%.2f  model-pick=%.2f %s  official-best=%.2f %s' % (
            case, q, n, s / base[2], s / mod[2], mod[0], s / off[2], off[0]))
        for params, pred, real in items:
            wins.setdefault((q, params), []).append(s / real / (s / off[2]))
    print('参数组合的平均相对最优比例（越接近 1 越好）：')
    for q in qs:
        ranked = sorted(((sum(v) / len(v), p) for (qq, p), v in wins.items() if qq == q), reverse=True)
        for score, p in ranked[:8]:
            print('  P%d %s  %.3f' % (q, p, score))
