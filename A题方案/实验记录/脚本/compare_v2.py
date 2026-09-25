"""results_v2 与 results_all 同一批 (用例, 问题, 核数) 的官方结果对比。"""
import collections
import csv
import statistics as st


def load(path):
    return {(r['case'], int(r['problem']), int(r['cores'])): r
            for r in csv.DictReader(open(path, encoding='utf-8-sig'))}


import sys
old, new = load('results_all/summary.csv'), load((sys.argv[1] if len(sys.argv) > 1 else 'results_v2') + '/summary.csv')
keys = sorted(k for k in new if k in old)
print('对比行数', len(keys))
for q in (1, 2, 3):
    for n in (2, 3, 4, 5):
        ks = [k for k in keys if k[1] == q and k[2] == n]
        if not ks:
            continue
        so = st.mean(float(old[k]['speedup']) for k in ks)
        sn = st.mean(float(new[k]['speedup']) for k in ks)
        print('P%d %d核: 平均加速比 %.3f -> %.3f' % (q, n, so, sn))
ratios = []
for k in keys:
    if k[2] == 1:
        continue
    a, b = float(old[k]['makespan']), float(new[k]['makespan'])
    ratios.append(((b - a) / a, k, old[k]['params'], new[k]['params']))
ratios.sort()
better = [r for r in ratios if r[0] < -0.005]
worse = [r for r in ratios if r[0] > 0.005]
print('多核行 %d：变好 %d，变差 %d，持平 %d' % (len(ratios), len(better), len(worse), len(ratios) - len(better) - len(worse)))
print('最大改进:')
for r in ratios[:12]:
    print('   %s P%d N%d %+.1f%%  %s -> %s' % (r[1][0], r[1][1], r[1][2], 100 * r[0], r[2], r[3]))
print('退化:')
for r in worse[::-1][:15]:
    print('   %s P%d N%d %+.1f%%  %s -> %s' % (r[1][0], r[1][1], r[1][2], 100 * r[0], r[2], r[3]))
flags = collections.Counter()
for k in keys:
    p = new[k]['params']
    for f in ('mc', 'blc', 'child'):
        if "'%s'" % f in p:
            flags[(k[1], f)] += 1
print('新开关被选中次数（含 1 核）:', dict(flags))
