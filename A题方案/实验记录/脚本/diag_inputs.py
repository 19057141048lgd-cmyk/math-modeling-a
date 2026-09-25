"""图输入的共享结构：每个输入的大小、消费者数、消费者跨多少个作业、在拓扑序中的跨度。"""
import sys
from collections import Counter, defaultdict

sys.path.insert(0, '.')
from solver.config import case_path
from solver.graph import Graph

for case in sys.argv[1:]:
    G = Graph(case_path(case))
    jobs = G.jobs()
    job_of = {u: j for j, ms in enumerate(jobs) for u in ms}
    pos = {u: i for i, u in enumerate(G.topo)}
    cons = defaultdict(list)
    size = {}
    for u in G.compute:
        for tid, s in G.inputs[u].items():
            cons[tid].append(u)
            size[tid] = s
    shared = {t: cs for t, cs in cons.items() if len(cs) > 1}
    big = sorted(shared, key=lambda t: -size[t] * len(shared[t]))[:5]
    print('%s: 作业=%d(最大作业占比 %.2f) 图输入=%d 总字节=%d 被多算子共享=%d(字节占比 %.2f)' % (
        case, len(jobs), sum(G.cycles[u] for u in jobs[0]) / G.total_cycles, len(cons), sum(size.values()),
        len(shared), sum(size[t] for t in shared) / max(sum(size.values()), 1)))
    print('   消费者数分布:', sorted(Counter(len(c) for c in cons.values()).items())[:12])
    for t in big:
        cs = shared[t]
        print('   输入%d: %dB x %d 个消费者, 跨作业 %d 个, 拓扑序跨度 %.2f' % (
            t, size[t], len(cs), len({job_of[u] for u in cs}),
            (max(pos[u] for u in cs) - min(pos[u] for u in cs)) / len(G.topo)))
