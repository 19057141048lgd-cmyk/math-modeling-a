"""复审 6 个问题在当前版本（results_all 已替换为新全量）上的存在情况。"""
import csv
import glob
import json
import os
import statistics as st
import sys

os.chdir(r'C:\Users\richd\Desktop\数学建模\A题(1)\A题方案')
sys.path.insert(0, '.')
rows = list(csv.DictReader(open('results_all/summary.csv', encoding='utf-8-sig')))
by = {(r['case'], int(r['problem']), int(r['cores'])): r for r in rows}
cases = sorted({r['case'] for r in rows})

print('== 问题1：第三问 T_noL2 / T_Cache（逐例求比再平均）')
for n in range(1, 6):
    rs = [float(by[(c, 2, n)]['makespan']) / float(by[(c, 3, n)]['makespan']) for c in cases]
    print('  %d核 平均=%.6f  <1 的用例数=%d  最大=%.4f' % (n, st.mean(rs), sum(x < 1 for x in rs), max(rs)))
print('  case_044 1核: 无L2=%s Cache=%s' % (by[('case_044', 2, 1)]['makespan'], by[('case_044', 3, 1)]['makespan']))
print('  table_p3.md 行数(附录):', sum(1 for l in open('results_all/table_p3.md', encoding='utf-8') if l.startswith('| case_')))

print('== 问题3：rows 缓存单核 speedup 与 CSV 不一致')
bad = 0
for p in glob.glob('results_all/rows/*.json'):
    for r in json.load(open(p, encoding='utf-8'))['rows']:
        if r['cores'] == 1 and abs(r['speedup'] - 1.0) > 1e-9:
            bad += 1
print('  不一致行数:', bad)

print('== 问题4：Cache 慢于无 L2 的反例')
for (c, q, n), r in sorted(by.items()):
    if q == 3 and float(r['makespan']) > float(by[(c, 2, n)]['makespan']):
        print('  %s %d核 无L2=%s Cache=%s 参数=%s' % (c, n, by[(c, 2, n)]['makespan'], r['makespan'], r['params']))

print('== 问题5：预测误差（剔除兜底替换的组合）')
for q in (1, 2, 3):
    rs = [r for r in rows if int(r['problem']) == q and r['makespan'] == r['pick_makespan']
          and not r['params'].startswith(('prev', 'p2_plan'))]
    e = [(float(r['predicted']) - float(r['makespan'])) / float(r['makespan']) for r in rs]
    print('  P%d 组合=%d 平均偏差=%+.3f%% MAE=%.3f%% 低估>10%%=%d' % (
        q, len(e), 100 * st.mean(e), 100 * st.mean(abs(x) for x in e), sum(x < -0.10 for x in e)))
for n in (4, 5):
    r = by[('case_035', 2, n)]
    print('  case_035 P2 %d核: 预测=%.0f 官方=%s 参数=%s' % (n, float(r['predicted']), r['makespan'], r['params']))
print('  P2 5核有 spill 的用例:', sum(int(by[(c, 2, 5)]['spill_added_copy_bytes'] or 0) > 0 for c in cases))

print('== 问题2：Cache 代理模型最小复现')
from solver.sched import CacheModel
c = CacheModel(1048576)
c.insert('A', 350000, 0)
c.insert('B', 350000, 10)
before = c.present('A', 50)
c.insert('C', 350000, 100)
print('  50 时刻查询 A：插入 C 之前=%s，插入 C（100 时刻）之后=%s' % (before, c.present('A', 50)))
