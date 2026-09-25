"""快速实验：拆开所有小作业（split_ratio=0）+ 细粒度，看 spill 与官方 makespan。"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, '.')
from solver.config import case_path, load_config
from solver.evaluate import run_official
from solver.graph import Graph
from solver.solve import build_candidate

hw = load_config()
q, n = int(sys.argv[1]), int(sys.argv[2])
scene = {1: 'A', 2: 'B', 3: 'C'}[q]
import csv
rows = {(r['case'], r['problem'], r['cores']): r
        for r in csv.DictReader(open('results_all/summary.csv', encoding='utf-8-sig'))}
for case in sys.argv[3:]:
    G = Graph(case_path(case))
    cur = rows[(case, str(q), str(n))]
    print('%s P%d N=%d 现有: makespan=%s spill=%s params=%s' % (
        case, q, n, cur['makespan'], cur['spill_added_copy_bytes'], cur['params']))
    res = []
    for grain in (64, 256, 1024, 4096):
        for loc in (0.0, 2.0, 8.0):
            t0 = time.time()
            plan, pred, n_sub, n_cl, spill = build_candidate(G, n, hw, scene, 0.0, grain, loc)
            res.append((pred, grain, loc, plan, spill, n_cl, n_sub, time.time() - t0))
    res.sort(key=lambda r: r[0])
    for pred, grain, loc, plan, spill, n_cl, n_sub, dt in res[:3]:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'plan.json')
            json.dump(plan, open(p, 'w'))
            out = run_official(case, q, p, tmp, tag='x')
        print('   grain=%d loc=%.0f 簇=%d 子图=%d 预测=%.0f spill预测=%d | 官方 makespan=%d spill=%d (%.1fs)' % (
            grain, loc, n_cl, n_sub, pred, spill, out['makespan'], out.get('spill_added_copy_bytes', -1), dt))
