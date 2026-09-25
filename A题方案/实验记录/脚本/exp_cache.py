"""问题3：旧 / 新 Cache 代理模型的排序质量对比。

python exp_cache.py <输出目录> [核数...]
对 results_val23 中的用例、每个核数：全部问题3 候选分别用旧模型和新模型生成方案与预测，
全部交官方评估（按方案内容去重），记录 (模型, 参数, 预测, 官方)。
"""
import csv
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = r'C:\Users\richd\Desktop\数学建模\A题(1)\A题方案'
os.chdir(ROOT)
sys.path.insert(0, ROOT)


class OldCacheModel:
    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = {}
        self.cum = 0

    def present(self, tid, t):
        e = self.entries.get(tid)
        return e is not None and e[0] <= t and self.cum - e[1] <= self.capacity

    def insert(self, tid, size, t):
        if size > self.capacity:
            return
        e = self.entries.get(tid)
        if e is not None and self.cum - e[1] <= self.capacity:
            return
        self.entries[tid] = (t, self.cum)
        self.cum += size


def old_load(self, tid, size, start, commit):
    if self.cache is not None and self.cache.present(tid, start):
        return start + size / self.hw.cache_bandwidth, True
    end = self.ddr.transfer(start, size, commit)
    if commit and self.cache is not None:
        self.cache.insert(tid, size, end)
    return end, False


def job(case, n, out):
    from solver import sched
    from solver.config import case_path, load_config
    from solver.evaluate import run_official
    from solver.graph import Graph
    from solver.solve import build_candidate, candidate_params, plan_signature
    path = os.path.join(out, '%s_n%d.json' % (case, n))
    if os.path.exists(path):
        return path
    hw = load_config()
    G = Graph(case_path(case))
    new_cls, new_load = sched.CacheModel, sched.ListScheduler._load
    # 本实验只比较 Cache 模型，关闭之后加入的 MTE3 顺序模型，与已完成的 42 组同口径
    sched.ListScheduler._mte3_ready = lambda self, k, f, among: f
    rec, done = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for model in ('old', 'new'):
            sched.CacheModel = OldCacheModel if model == 'old' else new_cls
            sched.ListScheduler._load = old_load if model == 'old' else new_load
            for params in candidate_params(G, n, 3, hw):
                plan, pred, n_sub, _, _ = build_candidate(G, n, hw, *params)
                sig = plan_signature(plan)
                if sig not in done:
                    p = os.path.join(tmp, 'plan.json')
                    with open(p, 'w', encoding='utf-8') as f:
                        json.dump(plan, f)
                    done[sig] = run_official(case, 3, p, tmp, tag='x')['makespan']
                rec.append({'model': model, 'params': str(params), 'scene': params[0],
                            'pred': pred, 'official': done[sig]})
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'case': case, 'cores': n, 'rows': rec}, f)
    return path


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    out = os.path.abspath(sys.argv[1])
    cores = [int(x) for x in sys.argv[2:]] or [3, 5]
    os.makedirs(out, exist_ok=True)
    cases = sorted({r['case'] for r in csv.DictReader(open('results_val23/summary.csv', encoding='utf-8-sig'))})
    from solver.config import case_path
    jobs = sorted(((c, n) for c in cases for n in cores), key=lambda x: -os.path.getsize(case_path(x[0])))
    with ProcessPoolExecutor(max_workers=int(os.environ.get('WORKERS', 8))) as ex:
        futs = {ex.submit(job, c, n, out): (c, n) for c, n in jobs}
        for fut in as_completed(futs):
            try:
                print('完成', futs[fut], fut.result(), flush=True)
            except Exception as e:
                print('失败', futs[fut], e, flush=True)
