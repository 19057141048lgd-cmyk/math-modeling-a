"""case_035 问题2：模型时间线与官方时间线逐段对比。python diag_035.py [核数] [参数...]"""
import json
import os
import sys
from collections import defaultdict

ROOT = r'C:\Users\richd\Desktop\数学建模\A题(1)\A题方案'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding='utf-8')
from solver.config import case_path, load_config
from solver.evaluate import run_official
from solver.graph import Graph, build_clusters
from solver.plan import plan_scene_b
from solver.sched import schedule
from solver.solve import predict

case = os.environ.get('CASE', 'case_035')
n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
params = eval(sys.argv[2]) if len(sys.argv) > 2 else ('B', 1.0, 64, 0.0)
hw = load_config()
G = Graph(case_path(case))
scene, split, grain, loc = params[:4]
clusters, _ = build_clusters(G, n, split, grain)
S = schedule(G, clusters, n, scene, hw, loc)
plan = plan_scene_b(clusters, S, G.total_cycles / (n * 40.0))
pred, spill = predict(G, plan, S, scene, hw)
out = os.path.join(os.environ['TEMP'], 'mm_extract', 'diag035')
os.makedirs(out, exist_ok=True)
pp = os.path.join(out, 'plan.json')
json.dump(plan, open(pp, 'w'))
res = run_official(case, 2, pp, out, tag='d', keep_trace=True)
print('模型 makespan=%.0f 预测=%.0f spill=%d | 官方=%d  簇=%d 子图=%d' % (
    S.makespan, pred, spill, res['makespan'], len(clusters), len(set(plan['node_to_subgraph'].values()))))

trace = json.load(open(os.path.join(out, '%s_d.trace.json' % case), encoding='utf-8'))
main = json.load(open(os.path.join(out, '%s_d.json' % case), encoding='utf-8'))
print('trace keys:', list(trace)[:20] if isinstance(trace, dict) else type(trace))
print('result keys:', list(main)[:30])
tl = trace.get('per_core_timeline') or main.get('per_core_timeline')
xf = trace.get('transfer_timeline') or main.get('transfer_timeline')

# 模型：每核各簇
model_core = defaultdict(lambda: [0.0, 0.0, 0.0])
for c in clusters:
    k = S.core[c.cid]
    model_core[k][0] += c.M
    model_core[k][1] += c.V
for k in range(n):
    print('模型 核%d: M=%.0f V=%.0f end=%.0f' % (k, model_core[k][0], model_core[k][1], S.core_end[k]))

for core in tl:
    busy = defaultdict(float)
    cnt = defaultdict(int)
    for e in core['ops']:
        busy[(e['pipe'], e['op'] if e['op'] in ('COPY_IN', 'COPY_OUT') else 'compute')] += e['duration']
        cnt[e['op'] if e['op'] in ('COPY_IN', 'COPY_OUT') else 'compute'] += 1
    end = max(e['end'] for e in core['ops'])
    print('官方 核%d end=%d ' % (core['core_id'], end) + ' '.join(
        '%s/%s=%.0f' % (p, o, v) for (p, o), v in sorted(busy.items())), dict(cnt))

# 跨核传输：release 到实际开始的等待（MTE2 队头阻塞）
if xf:
    waits = [x['copy_in_start'] - x['copy_in_release'] for x in xf]
    waits.sort()
    print('跨核传输 %d 条；COPY_IN 发射晚于可发射时刻的等待：中位 %.0f，P90 %.0f，最大 %.0f，总计 %.0f' % (
        len(xf), waits[len(waits) // 2], waits[int(len(waits) * 0.9)], waits[-1], sum(waits)))
    print('样例 transfer:', {k: xf[0][k] for k in list(xf[0])[:14]})

# 模型簇完成时刻 vs 官方子图完成时刻（按子图）
sg_of_cl = {c.cid: plan['node_to_subgraph'][str(c.ops[0])] for c in clusters}
model_sg = defaultdict(lambda: [1e18, 0.0])
for c in clusters:
    s = model_sg[sg_of_cl[c.cid]]
    s[0] = min(s[0], S.start[c.cid])
    s[1] = max(s[1], S.fin[c.cid])
off_sg = {}
for core in tl:
    for e in core['subgraphs']:
        off_sg[e['subgraph_id']] = (e['start'], e['end'], core['core_id'])
lag = sorted(((off_sg[sg][1] - model_sg[sg][1], sg) for sg in off_sg), reverse=True)
print('子图完成时刻 官方-模型：最大 %s' % lag[:5])
order = sorted(off_sg, key=lambda s: off_sg[s][1])
print('按官方完成时刻采样（子图, 核, 模型结束, 官方结束）：')
for sg in order[::max(1, len(order) // 25)]:
    print('  sg=%4d 核%d 模型=%8.0f 官方=%8d 差=%7.0f' % (sg, off_sg[sg][2], model_sg[sg][1], off_sg[sg][1],
                                                     off_sg[sg][1] - model_sg[sg][1]))
