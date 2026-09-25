import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

SOL = r'C:\Users\richd\Desktop\数学建模\A题(1)\A题方案'
sys.path.insert(0, SOL)
from solver.config import case_path  # noqa: E402
from solver.graph import Graph  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
out = os.path.join(SOL, sys.argv[1] if len(sys.argv) > 1 else 'results_all')
with open(os.path.join(out, 'summary.csv'), encoding='utf-8-sig') as f:
    rows = {(r['case'], int(r['problem']), int(r['cores'])): r for r in csv.DictReader(f)}
cases = sorted({k[0] for k in rows})
single = {c: float(rows[(c, 1, 1)]['single_makespan']) for c in cases}


def ms(k):
    return float(rows[k]['makespan'])


def q4(v):
    v = sorted(v)
    return v[0], v[len(v) // 4], statistics.median(v), v[3 * len(v) // 4], v[-1]


print('用例 %d 个，组合 %d 个' % (len(cases), len(rows)))
print('\n== 平均加速比（中位数）')
for q in (1, 2, 3):
    print('P%d  ' % q + '  '.join('%d核 %.3f (%.2f)' % (n, statistics.mean(single[c] / ms((c, q, n)) for c in cases),
                                                   statistics.median(single[c] / ms((c, q, n)) for c in cases))
                                  for n in range(1, 6)))
print('\n== 5 核加速比分布（最小/25%/中位/75%/最大）')
for q in (1, 2, 3):
    print('P%d  %s' % (q, ' / '.join('%.2f' % x for x in q4([single[c] / ms((c, q, 5)) for c in cases]))))
sp5 = [single[c] / ms((c, 2, 5)) for c in cases]
print('P2 5核：>=4.5 %d 个，>=4 %d 个，<2 %d 个，超线性(>5) %d 个' % (
    sum(x >= 4.5 for x in sp5), sum(x >= 4 for x in sp5), sum(x < 2 for x in sp5), sum(x > 5 for x in sp5)))
sup = [(c, q, n) for (c, q, n) in rows if n > 1 and single[c] / ms((c, q, n)) > n]
print('超线性组合 %d 个（%s）' % (len(sup), Counter(c for c, _, _ in sup).most_common(6)))

print('\n== P3/P2（只读 Cache 相对无 L2）')
for n in range(1, 6):
    v = {c: ms((c, 2, n)) / ms((c, 3, n)) for c in cases}
    top = max(v, key=v.get)
    print('%d核 平均 %.3f 中位 %.3f 最高 %.3f(%s) >1.01 的 %d 个 =1 的 %d 个' % (
        n, statistics.mean(v.values()), statistics.median(v.values()), v[top], top,
        sum(x > 1.01 for x in v.values()), sum(abs(x - 1) < 1e-9 for x in v.values())))

print('\n== 单调性与兜底')
bad = sum(1 for (c, q, n) in rows if n > 1 and ms((c, q, n)) > ms((c, q, n - 1)))
bad3 = sum(1 for (c, q, n) in rows if q == 3 and ms((c, q, n)) > ms((c, 2, n)))
fb = [(k, r['params']) for k, r in rows.items() if r.get('pick_makespan') and float(r['pick_makespan']) > ms(k)]
print('核数不单调 %d  P3劣于P2 %d  兜底 %d 次：prev %d / p2 %d' % (
    bad, bad3, len(fb), sum(p.startswith('prev') for _, p in fb), sum(p == 'p2_plan' for _, p in fb)))
gain = [float(rows[k]['pick_makespan']) / ms(k) for k, _ in fb]
if gain:
    print('兜底挽回：平均 %.3f 最大 %.3f' % (statistics.mean(gain), max(gain)))

print('\n== 耗时')
st = {k: float(r['solve_seconds']) for k, r in rows.items()}
et = {k: float(r['eval_seconds']) for k, r in rows.items()}
k1 = max(st, key=st.get)
k2 = max(et, key=et.get)
print('求解：平均 %.1f s，中位 %.1f s，最长 %.1f s（%s P%d N=%d）' % (
    statistics.mean(st.values()), statistics.median(st.values()), st[k1], *k1))
print('评估：合计 %.0f s，最长 %.0f s（%s P%d N=%d）' % (sum(et.values()), et[k2], *k2))
per_case = {c: sum(st[k] + et[k] for k in rows if k[0] == c) for c in cases}
slow = sorted(per_case, key=per_case.get, reverse=True)[:8]
print('单用例总耗时最长：' + '，'.join('%s %.0f s' % (c, per_case[c]) for c in slow))
solve_case = {c: max(st[k] for k in rows if k[0] == c) for c in cases}
print('单组合求解超过 60 s 的用例：%s' % sorted((c, round(v)) for c, v in solve_case.items() if v > 60))

print('\n== spill')
sp_single = {}
for c in cases:
    p = os.path.join(SOL, 'results', 'single', '%s_single.json' % c)
    with open(p, encoding='utf-8') as f:
        sp_single[c] = json.load(f)['data_movement_bytes'].get('spill_added_copy_bytes', 0)
print('官方单核基线有 spill 的用例 %d 个' % sum(v > 0 for v in sp_single.values()))
for q in (1, 2, 3):
    print('P%d 选中方案有 spill 的用例数：' % q + '  '.join(
        '%d核 %d' % (n, sum(float(rows[(c, q, n)]['spill_added_copy_bytes']) > 0 for c in cases)) for n in range(1, 6)))

print('\n== 与下界 LB = max(关键路径, M/N, V/N, IO/60) 的差距')
gap = defaultdict(dict)
bind = defaultdict(Counter)
prof = {}
for c in cases:
    G = Graph(case_path(c))
    M = sum(G.cycles[v] for v in G.compute if G.is_m[v])
    V = G.total_cycles - M
    cp = {}
    for v in G.topo:
        cp[v] = G.cycles[v] + max((cp[p] for p in G.pred[v]), default=0)
    CP = max(cp.values())
    uniq = {}
    for v in G.compute:
        uniq.update(G.inputs[v])
    io = (sum(uniq.values()) + sum(G.out_bytes.values())) / 60.0
    jobs = G.jobs()
    prof[c] = (len(G.compute), G.total_cycles, G.total_cycles / CP,
               sum(G.cycles[v] for v in jobs[0]) / G.total_cycles)
    for n in range(1, 6):
        parts = {'关键路径': CP, 'M/N': M / n, 'V/N': V / n, 'IO/60': io}
        lb = max(parts.values())
        bind[n][max(parts, key=parts.get)] += 1
        for q in (1, 2, 3):
            gap[(q, n)][c] = ms((c, q, n)) / lb
for q in (1, 2, 3):
    print('P%d  ' % q + '  '.join('%d核 %.2f' % (n, statistics.mean(gap[(q, n)].values())) for n in range(1, 6)))
for q in (1, 2, 3):
    v = list(gap[(q, 5)].values())
    print('P%d 5核：1.1 倍以内 %d 个，1.25 倍以内 %d 个，超过 2 倍 %d 个' % (
        q, sum(x <= 1.1 for x in v), sum(x <= 1.25 for x in v), sum(x > 2 for x in v)))
print('起作用的下界项：' + '；'.join('%d核 %s' % (n, dict(bind[n])) for n in (1, 4, 5)))
worst = sorted(gap[(2, 5)], key=gap[(2, 5)].get, reverse=True)[:8]
print('P2 5核离下界最远：' + '，'.join('%s %.2f(算子 %d, 并行度 %.1f)' % (c, gap[(2, 5)][c], prof[c][0], prof[c][2])
                                    for c in worst))


def group(label, cs):
    if not cs:
        return
    print('%s：%d 个用例；5核平均加速比 P1 %.2f / P2 %.2f / P3 %.2f；P2 5核 makespan/LB %.2f；P3/P2 5核 %.3f' % (
        label, len(cs), *(statistics.mean(single[c] / ms((c, q, 5)) for c in cs) for q in (1, 2, 3)),
        statistics.mean(gap[(2, 5)][c] for c in cs), statistics.mean(ms((c, 2, 5)) / ms((c, 3, 5)) for c in cs)))


print('\n== 分组')
for lo, hi in ((0, 10), (10, 43), (43, 1e9)):
    group('并行度 [%g, %g)' % (lo, hi), [c for c in cases if lo <= prof[c][2] < hi])
group('最大作业 > 50%', [c for c in cases if prof[c][3] > 0.5])
group('最大作业 <= 50%', [c for c in cases if prof[c][3] <= 0.5])
group('单核基线有 spill', [c for c in cases if sp_single[c] > 0])
group('单核基线无 spill', [c for c in cases if sp_single[c] == 0])
cs = [c for c in cases if sp_single[c] > 0]
if cs:
    print('单核基线有 spill 的用例：P2 1核平均加速比 %.3f，P3 1核 %.3f' % (
        statistics.mean(single[c] / ms((c, 2, 1)) for c in cs), statistics.mean(single[c] / ms((c, 3, 1)) for c in cs)))
by_size = sorted(cases, key=lambda c: prof[c][0])
for lo, hi in ((0, 2000), (2000, 10000), (10000, 1e9)):
    cs = [c for c in cases if lo <= prof[c][0] < hi]
    if cs:
        print('算子数 [%g, %g)：%d 个用例，P2 5核平均加速比 %.2f，单组合求解平均 %.1f s / 最长 %.1f s' % (
            lo, hi, len(cs), statistics.mean(single[c] / ms((c, 2, 5)) for c in cs),
            statistics.mean(st[k] for k in rows if k[0] in cs), max(st[k] for k in rows if k[0] in cs)))
