"""作业按共享图输入分组：并查集连接共用输入的作业，统计每组的作业数、共享输入字节和每作业用到的共享输入字节。"""
import sys
from collections import defaultdict

sys.path.insert(0, '.')
from solver.config import case_path
from solver.graph import Graph

for case in sys.argv[1:]:
    G = Graph(case_path(case))
    jobs = G.jobs()
    job_of = {u: j for j, ms in enumerate(jobs) for u in ms}
    users = defaultdict(set)
    size = {}
    for u in G.compute:
        for t, s in G.inputs[u].items():
            users[t].add(job_of[u])
            size[t] = s
    parent = list(range(len(jobs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for t, js in users.items():
        js = sorted(js)
        for j in js[1:]:
            parent[find(j)] = find(js[0])
    groups = defaultdict(list)
    for j in range(len(jobs)):
        groups[find(j)].append(j)
    per_job = defaultdict(int)
    for t, js in users.items():
        if len(js) > 1:
            for j in js:
                per_job[j] += size[t]
    gs = sorted(groups.values(), key=len, reverse=True)
    print('%s: 作业=%d 共享分组=%d 最大组作业数=%d  每作业共享输入字节 中位=%d 最大=%d' % (
        case, len(jobs), len(gs), len(gs[0]),
        sorted(per_job.values())[len(per_job) // 2] if per_job else 0, max(per_job.values(), default=0)))
    # 两个作业共享输入的重叠程度：取第一个作业，看与它共享输入最多的作业们
    j0 = gs[0][0]
    ov = defaultdict(int)
    for t, js in users.items():
        if j0 in js:
            for j in js:
                if j != j0:
                    ov[j] += size[t]
    top = sorted(ov.values(), reverse=True)
    print('   作业%d 与其他作业共享字节: 前5=%s 共享伙伴数=%d, 本作业共享输入=%d' % (j0, top[:5], len(ov), per_job[j0]))
