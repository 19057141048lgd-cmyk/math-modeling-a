"""诊断场景 B spill 的来源：单算子工作集是否超容量、峰值时驻留张量的类别与寿命。"""
import json
import sys
from collections import defaultdict

sys.path.insert(0, '.')
from solver.config import case_path, load_config
from solver.graph import Graph
from solver.memory import COPY_IN, estimate_spill, raw_view, step1_sequence, tasks_scene_b

hw = load_config()
cap = {'L1': hw.l1, 'UB': hw.ub}
cases = sys.argv[1:] or ['case_092', 'case_073', 'case_028', 'case_091', 'case_072']
for case in cases:
    G = Graph(case_path(case))
    R = raw_view(G.path)
    # 单算子工作集
    worst = defaultdict(int)
    for u in G.compute:
        ws = defaultdict(int)
        for t in R.touch[u]:
            ws[R.pos[t]] += R.size[t]
        for T in ws:
            worst[T] = max(worst[T], ws[T])
    plan = json.load(open('results_all/plans/%s_p2_n5.json' % case))
    total, per_core, ok = estimate_spill(G, plan, 'B', hw)
    single = {'node_to_subgraph': {str(u): 0 for u in G.compute}, 'core_schedules': [[0]]}
    s1, _, _ = estimate_spill(G, single, 'B', hw)
    print('%s: ops=%d  单算子最大工作集 L1=%d UB=%d (容量 %d/%d)  5核spill=%d  1核单子图spill=%d' % (
        case, len(G.compute), worst['L1'], worst['UB'], hw.l1, hw.ub, total, s1))
    tasks, rank = tasks_scene_b(R, plan)
    for task in tasks:
        if not per_core.get(task.core):
            continue
        seq = step1_sequence(task)
        seq.sort(key=lambda u: rank[task.sub[u]])
        step = {u: i for i, u in enumerate(seq)}
        first, last, cat = {}, {}, {}
        for u in seq:
            s = step[u]
            for t in list(task.outs.get(u, ())) + list(task.ins.get(u, ())):
                first.setdefault(t, s)
                last[t] = s
            for t in task.outs.get(u, ()):
                if task.kind[u] == COPY_IN:
                    cat[t] = 'ddr_in' if not R.producers.get(t) or R.kind[R.producers[t][0]] == 'COPY_IN' else 'xcore_in'
                else:
                    cat.setdefault(t, 'local')
        n = len(seq)
        for T in ('L1', 'UB'):
            prof = [0] * (n + 1)
            for t in first:
                if R.pos[t] == T:
                    prof[first[t]] += R.size[t]
                    prof[last[t] + 1] -= R.size[t]
            acc, peak, ps = 0, 0, 0
            for i in range(n):
                acc += prof[i]
                if acc > peak:
                    peak, ps = acc, i
            if peak <= cap[T]:
                continue
            br = defaultdict(int)
            span = defaultdict(int)
            for t in first:
                if R.pos[t] == T and first[t] <= ps <= last[t]:
                    br[cat.get(t, '?')] += R.size[t]
                    L = last[t] - first[t]
                    span['短(<20步)' if L < 20 else ('中(<200步)' if L < 200 else '长')] += R.size[t]
            print('   核%d %s: spill=%d 峰值=%d (%.1f倍容量) 序列长=%d 峰值组成=%s 寿命=%s' % (
                task.core, T, per_core[task.core], peak, peak / cap[T], n, dict(br), dict(span)))
