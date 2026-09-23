"""开发工具：spill 模型与官方 spill_added_copy_bytes 逐方案对比。

用法（在 A题方案 目录下）：python -m solver.dev_spill results results/dev/noverify
每个目录需含 summary.csv 与 plans/。
"""
import csv
import json
import os
import sys
import time

from .config import case_path, load_config
from .graph import Graph
from .memory import estimate_spill


def main(dirs):
    hw = load_config()
    graphs = {}
    exact = total = 0
    diffs = []
    t0 = time.time()
    for d in dirs:
        with open(os.path.join(d, 'summary.csv'), encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            case, q, n = r['case'], int(r['problem']), int(r['cores'])
            with open(os.path.join(d, 'plans', '%s_p%d_n%d.json' % (case, q, n)), encoding='utf-8') as f:
                plan = json.load(f)
            if case not in graphs:
                graphs[case] = Graph(case_path(case))
            est, _, ok = estimate_spill(graphs[case], plan, 'A' if q == 1 else 'B', hw)
            off = int(r['spill_added_copy_bytes'])
            total += 1
            exact += est == off and ok
            if est != off or not ok:
                diffs.append((abs(est - off), case, q, n, est, off, ok))
    print('方案 %d 个，spill 字节完全一致 %d 个（%.1fs）' % (total, exact, time.time() - t0))
    for item in sorted(diffs, reverse=True)[:15]:
        print('  差 %d: %s P%d N=%d 模型=%d 官方=%d 可行=%s' % item)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main(sys.argv[1:] or ['results'])
