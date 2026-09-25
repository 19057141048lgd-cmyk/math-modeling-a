import csv
import json
import os
import sys
from collections import defaultdict

ROOT = r'C:\Users\richd\Desktop\数学建模\A题(1)\A题方案'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding='utf-8')
from solver.config import case_path

rows = list(csv.DictReader(open('results_all/summary.csv', encoding='utf-8-sig')))
ev = defaultdict(float)
reg = defaultdict(list)
for r in rows:
    if int(r['cores']) > 1:
        ev[r['case']] += float(r['eval_seconds'])
old = {(r['case'], r['problem'], r['cores']): r for r in
       csv.DictReader(open('results_v1_对照/summary.csv', encoding='utf-8-sig'))}
for r in rows:
    o = old.get((r['case'], r['problem'], r['cores']))
    if o and int(r['cores']) > 1 and float(r['makespan']) > 1.005 * float(o['makespan']):
        reg[r['case']].append(float(r['makespan']) / float(o['makespan']) - 1)
nodes = {}
for c in ev:
    g = json.load(open(case_path(c), encoding='utf-8'))
    nodes[c] = len(g.get('nodes', g.get('ops', [])))
cs = sorted(ev, key=lambda c: nodes[c])
tot = sum(ev.values())
acc = 0
print('按节点数排序：用例 节点数 多核评估总秒数 累计占比 退化数')
for th in (500, 1000, 1500, 2000, 3000, 4000, 6000, 10**9):
    sub = [c for c in cs if nodes[c] <= th]
    print('节点数 <= %8d: 用例 %3d 个，评估耗时占比 %.1f%%，包含退化组合 %d 个 / 共 %d' % (
        th, len(sub), 100 * sum(ev[c] for c in sub) / tot, sum(len(reg[c]) for c in sub),
        sum(len(v) for v in reg.values())))
print('总多核评估秒数 %.0f' % tot)
