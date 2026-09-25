import json
import os
import sys
from collections import defaultdict

ROOT = r'C:\Users\richd\Desktop\数学建模\A题(1)\A题方案'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding='utf-8')
from solver.config import case_path, load_config
from solver.graph import Graph, build_clusters
from solver.plan import plan_scene_b
from solver.sched import schedule

case, n, params = 'case_035', 5, ('B', 1.0, 64, 0.0)
lo, hi = float(sys.argv[1]), float(sys.argv[2])
hw = load_config()
G = Graph(case_path(case))
clusters, _ = build_clusters(G, n, params[1], params[2])
S = schedule(G, clusters, n, 'B', hw, params[3])
plan = plan_scene_b(clusters, S, G.total_cycles / (n * 40.0))
out = os.path.join(os.environ['TEMP'], 'mm_extract', 'diag035')
main = json.load(open(os.path.join(out, '%s_d.json' % case), encoding='utf-8'))
sg_cl = defaultdict(list)
for c in clusters:
    sg_cl[plan['node_to_subgraph'][str(c.ops[0])]].append(c.cid)
for core in main['per_core_timeline']:
    busy = [e for e in core['subgraphs'] if e['end'] > lo and e['start'] < hi]
    if not busy:
        continue
    print('核%d 在 [%d,%d] 内的子图 %d 个：' % (core['core_id'], lo, hi, len(busy)))
    for e in busy[:40]:
        cids = sg_cl[e['subgraph_id']]
        ms = min(S.start[c] for c in cids)
        mf = max(S.fin[c] for c in cids)
        W = sum(clusters[c].M + clusters[c].V for c in cids)
        rem = sum(1 for c in cids for p in clusters[c].preds if S.core[p] != core['core_id'])
        print('  sg=%4d 官方 %6d-%6d 模型 %7.0f-%7.0f 模型核%d 计算量=%6.0f 跨核入边=%d' % (
            e['subgraph_id'], e['start'], e['end'], ms, mf, S.core[cids[0]], W, rem))
    ops = [o for o in core['ops'] if o['end'] > lo and o['start'] < hi]
    by = defaultdict(float)
    for o in ops:
        by[(o['pipe'], o['op'])] += min(o['end'], hi) - max(o['start'], lo)
    print('  区间内各 pipe 忙碌：', sorted(((k, int(v)) for k, v in by.items()), key=lambda x: -x[1])[:6])
