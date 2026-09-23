"""开发期统计：各用例的 DDR 压力与共享输入规模，用于挑选问题 3 的代表用例。"""
import glob
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from .config import data_dir
from .graph import Graph


def one(path):
    G = Graph(path)
    in_bytes = {}
    jobs_of = defaultdict(set)
    job_id = {}
    for j, members in enumerate(G.jobs()):
        for u in members:
            job_id[u] = j
    for u in G.compute:
        for tid, size in G.inputs[u].items():
            in_bytes[tid] = size
            jobs_of[tid].add(job_id[u])
    total_in = sum(in_bytes.values())
    shared = sum(in_bytes[t] for t, js in jobs_of.items() if len(js) > 1)
    out = sum(G.out_bytes.values())
    per_core5 = G.total_cycles / 5.0
    return (G.name, len(G.compute), (total_in + out) / 60.0 / per_core5, shared / max(total_in, 1),
            total_in, out)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    files = sorted(glob.glob(os.path.join(data_dir(), 'case_*.json')))
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(one, files))
    rows.sort(key=lambda r: -r[2])
    print('case      ops    DDR时间/5核单核计算  共享输入字节占比  输入字节  输出字节')
    for r in rows[:25]:
        print('%-9s %6d %10.2f %16.2f %10d %9d' % r)
