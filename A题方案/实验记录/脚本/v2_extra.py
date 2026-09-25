import csv
import json
import os
import statistics
import sys

SOL = r'C:\Users\richd\Desktop\数学建模\A题(1)\A题方案'
sys.stdout.reconfigure(encoding='utf-8')
with open(os.path.join(SOL, 'results_all', 'summary.csv'), encoding='utf-8-sig') as f:
    A = {(r['case'], int(r['problem']), int(r['cores'])): r for r in csv.DictReader(f)}
cases = sorted({k[0] for k in A})
ms = {k: float(r['makespan']) for k, r in A.items()}
single = {c: float(A[(c, 1, 1)]['single_makespan']) for c in cases}

print('== 1 核重切分：单核基准 / 1 核 makespan')
for q in (1, 2, 3):
    v = [single[c] / ms[(c, q, 1)] for c in cases]
    print('P%d 平均 %.3f  不等于1 %d  低于1 %d  最低 %.3f  最高 %.3f' % (
        q, statistics.mean(v), sum(abs(x - 1) > 1e-9 for x in v), sum(x < 1 - 1e-9 for x in v), min(v), max(v)))

print('\n== 兜底')
fb = [k for k, r in A.items() if float(r['pick_makespan']) > ms[k]]
red = [1 - ms[k] / float(A[k]['pick_makespan']) for k in fb]
print('次数 %d  prev %d  p2 %d  平均降低 %.2f%%  最大 %.2f%%' % (
    len(fb), sum(A[k]['params'].startswith('prev') for k in fb), sum(A[k]['params'] == 'p2_plan' for k in fb),
    100 * statistics.mean(red), 100 * max(red)))
pre = {}
for q in (1, 2, 3):
    bad = set()
    for c in cases:
        for n in range(2, 6):
            if float(A[(c, q, n)]['pick_makespan']) > float(A[(c, q, n - 1)]['pick_makespan']):
                bad.add(c)
    pre[q] = len(bad)
print('兜底前（pick）核数增加反而变慢的用例数：P1 %d P2 %d P3 %d' % (pre[1], pre[2], pre[3]))
bad3 = sum(1 for c in cases for n in range(1, 6) if float(A[(c, 3, n)]['pick_makespan']) > ms[(c, 2, n)])
print('兜底前 P3 选中方案劣于 P2 的组合 %d' % bad3)

print('\n== 预测精度（不含兜底行）')
for q in (1, 2, 3):
    e = [float(A[k]['predicted']) / ms[k] - 1 for k in A if k[1] == q and k not in fb]
    print('P%d 行数 %d 偏差 %+.2f%% MAE %.2f%% 低估>10%% %d' % (
        q, len(e), 100 * statistics.mean(e), 100 * statistics.mean(abs(x) for x in e), sum(x < -0.1 for x in e)))

print('\n== Cache 比值最高的组合（P2/P3）')
r = sorted(((ms[(c, 2, n)] / ms[(c, 3, n)], c, n) for c in cases for n in range(1, 6)), reverse=True)[:6]
print(r)

print('\n== P2 5 核超线性用例的单核 spill')
sp = {}
for c in cases:
    with open(os.path.join(SOL, 'results', 'single', '%s_single.json' % c), encoding='utf-8') as f:
        sp[c] = json.load(f)['data_movement_bytes'].get('spill_added_copy_bytes', 0)
sup = [c for c in cases if single[c] / ms[(c, 2, 5)] > 5]
print('共 %d 个，其中单核有 spill %d 个；无 spill 的：%s' % (len(sup), sum(sp[c] > 0 for c in sup),
                                                   [(c, round(single[c] / ms[(c, 2, 5)], 2)) for c in sup if sp[c] == 0]))
print('\n== 5 核：P1 快于 P2 的用例数 %d' % sum(ms[(c, 1, 5)] < ms[(c, 2, 5)] for c in cases))
print('== 16 代表用例 4 核 / 5 核平均加速比')
rep = ['case_%03d' % i for i in (1, 4, 10, 19, 26, 29, 44, 46, 48, 50, 57, 61, 64, 71, 80, 94)]
for q in (1, 2, 3):
    print('P%d  4核 %.3f  5核 %.3f' % (q, statistics.mean(single[c] / ms[(c, q, 4)] for c in rep),
                                     statistics.mean(single[c] / ms[(c, q, 5)] for c in rep)))
