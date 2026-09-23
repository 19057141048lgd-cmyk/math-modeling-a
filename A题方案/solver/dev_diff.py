"""开发期诊断：逐子图对比模型时间线与官方时间线（场景 B）。"""
import json
import os
import sys

from .config import SOLUTION_DIR, case_path, load_config
from .evaluate import run_official
from .graph import Graph, build_clusters
from .plan import plan_scene_b, validate_plan
from .sched import schedule

OUT = os.path.join(SOLUTION_DIR, 'results', 'dev')

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    case, n, split, grain = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    chunk_div = float(sys.argv[5]) if len(sys.argv) > 5 else 40.0
    G = Graph(case_path(case))
    clusters, _ = build_clusters(G, n, split, grain)
    S = schedule(G, clusters, n, 'B', load_config())
    plan = plan_scene_b(clusters, S, G.total_cycles / (n * chunk_div))
    validate_plan(G, plan)
    path = os.path.join(OUT, '%s_diff_plan.json' % case)
    json.dump(plan, open(path, 'w', encoding='utf-8'))
    r = run_official(case, 2, path, OUT, tag='diff')
    off = json.load(open(os.path.join(OUT, '%s_diff.json' % case), encoding='utf-8'))
    op_sg = {int(k): v for k, v in plan['node_to_subgraph'].items()}
    model = {}
    for c in clusters:
        sg = op_sg[c.ops[0]]
        s, f = model.get(sg, (1e18, 0))
        model[sg] = (min(s, S.start[c.cid]), max(f, S.fin[c.cid]))
    print('predicted=%.0f official=%d  clusters=%d subgraphs=%d' % (
        S.makespan, r['makespan'], len(clusters), len(model)))
    for core in off['per_core_timeline']:
        k = core['core_id']
        rows = core['subgraphs']
        print('core %d: official end=%d, model end=%.0f' % (
            k, max(x['end'] for x in rows), max(model[x['subgraph_id']][1] for x in rows)))
        for x in rows[:: max(1, len(rows) // 12)]:
            ms, mf = model[x['subgraph_id']]
            print('   sg %4d  official [%6d, %6d]  model [%6.0f, %6.0f]  lag=%+6.0f' % (
                x['subgraph_id'], x['start'], x['end'], ms, mf, x['end'] - mf))
    tr = off['cross_core_transfers']
    waits = sorted(((t['copy_in_start'] - t['copy_in_release']), t['target_core']) for t in tr)
    print('cross-core transfers=%d, copy_in start-release lag: median=%d p90=%d max=%d' % (
        len(tr), waits[len(waits) // 2][0], waits[int(len(waits) * 0.9)][0], waits[-1][0]))
