"""计算图载入与预处理：计算算子 DAG、独立作业分解、依赖树聚簇。"""
import json
import os
from collections import defaultdict

COPY_TYPES = ('COPY_IN', 'COPY_OUT')


class Graph:
    """算子-张量二部图的计算算子视图（COPY_IN/COPY_OUT 收缩为输入/输出属性）。"""

    def __init__(self, path):
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
        ops = {o['id']: o for o in raw['ops']}
        tensors = {t['id']: t for t in raw['tensors']}
        producer, consumers = {}, defaultdict(list)
        for e in raw['edges']:
            s, t = e['source'], e['target']
            if s in ops and t in tensors:
                producer[t] = s
            elif s in tensors and t in ops:
                consumers[s].append(t)

        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.compute = sorted(i for i, o in ops.items() if o['op'] not in COPY_TYPES)
        cset = set(self.compute)
        self.cycles = {i: ops[i]['cycles'] for i in self.compute}
        self.is_m = {i: ops[i]['pipe'] == 'PIPE_M' for i in self.compute}
        self.succ = {i: set() for i in self.compute}
        self.pred = {i: set() for i in self.compute}
        self.out_tensors = {i: [] for i in self.compute}   # 该算子产出、被其他计算算子消费的中间张量
        self.inputs = {i: {} for i in self.compute}        # 图输入（经原 COPY_IN 搬入）的片上张量 -> 字节
        self.op_inputs = {i: [] for i in self.compute}     # (张量, 字节, 生产计算算子或 -1 表示图输入)
        self.out_bytes = {i: 0 for i in self.compute}      # 图输出（经原 COPY_OUT 写回）的字节
        self.tensor_size = {}
        self.tensor_consumers = {}

        for tid, cs in consumers.items():
            p = producer.get(tid)
            if p is None:
                continue
            size = tensors[tid]['size']
            if p in cset:
                inner = [c for c in cs if c in cset and c != p]
                if len(inner) < len(cs):
                    self.out_bytes[p] += size
                if inner:
                    self.out_tensors[p].append(tid)
                    self.tensor_size[tid] = size
                    self.tensor_consumers[tid] = inner
                    for c in inner:
                        self.succ[p].add(c)
                        self.pred[c].add(p)
                        self.op_inputs[c].append((tid, size, p))
            elif ops[p]['op'] == 'COPY_IN':
                for c in cs:
                    if c in cset:
                        self.inputs[c][tid] = size
                        self.op_inputs[c].append((tid, size, -1))

        self.topo = self._topo_order()
        self.depth = {}
        finish = {}
        for u in self.topo:
            self.depth[u] = 1 + max((self.depth[p] for p in self.pred[u]), default=0)
            finish[u] = self.cycles[u] + max((finish[p] for p in self.pred[u]), default=0)
        self.total_cycles = sum(self.cycles.values())
        self.parallelism = self.total_cycles / max(max(finish.values()), 1)   # 总计算量 / 关键路径

    def _topo_order(self):
        indeg = {i: len(self.pred[i]) for i in self.compute}
        order = [i for i in self.compute if indeg[i] == 0]
        k = 0
        while k < len(order):
            u = order[k]
            k += 1
            for v in sorted(self.succ[u]):
                indeg[v] -= 1
                if indeg[v] == 0:
                    order.append(v)
        if len(order) != len(self.compute):
            raise ValueError('计算图存在环')
        return order

    def jobs(self):
        """按中间张量连通性划分的独立作业（弱连通分量），作业之间没有任何依赖。"""
        parent = {i: i for i in self.compute}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for u in self.compute:
            for v in self.succ[u]:
                ru, rv = find(u), find(v)
                if ru != rv:
                    parent[ru] = rv
        groups = defaultdict(list)
        for u in self.topo:
            groups[find(u)].append(u)
        return sorted(groups.values(), key=lambda ms: -sum(self.cycles[i] for i in ms))


class Cluster:
    __slots__ = ('cid', 'ops', 'M', 'V', 'cp', 'solo', 'preds', 'succs',
                 'inputs', 'out_bytes', 'job', 'plan')

    def __init__(self, cid, job):
        self.cid = cid
        self.job = job
        self.ops = []
        self.preds = {}    # 前驱簇 -> 需要从该簇接收的中间张量列表
        self.succs = set()
        self.inputs = {}
        self.out_bytes = 0
        # 核内执行计划：按官方 step1 同规则的逆向 DFS 序，每项为
        # (是否 PIPE_M, cycles, 簇内前驱在 plan 中的下标, 外部输入 ((张量, 字节, 生产簇或 -1), ...))
        self.plan = []


def build_clusters(G, n_cores, split_ratio=0.5, grain=16):
    """两级粗化。

    1. 独立作业：计算量不超过 split_ratio * W / N 的作业整体作为一个簇（不可拆）；
    2. 大作业：按“单消费者依赖树”自底向上合并（in-tree 聚簇），簇计算量上限 W / (N * grain)。
       in-tree 簇只有根算子向外输出，可以证明簇图必然无环。
    """
    W = G.total_cycles
    big = split_ratio * W / n_cores
    cap = max(W / (n_cores * grain), max(G.cycles.values()))
    root = {}
    clusters = []
    cluster_of = {}
    for j, members in enumerate(G.jobs()):
        weight = sum(G.cycles[i] for i in members)
        if weight <= big:
            c = Cluster(len(clusters), j)
            clusters.append(c)
            for u in members:
                cluster_of[u] = c.cid
            continue
        mset = set(members)
        wt = {}
        for u in reversed(members):
            s = G.succ[u]
            if len(s) == 1 and G.out_bytes[u] == 0:
                v = next(iter(s))
                r = root[v]
                if wt[r] + G.cycles[u] <= cap:
                    root[u] = r
                    wt[r] += G.cycles[u]
                    continue
            root[u] = u
            wt[u] = G.cycles[u]
        local = {}
        for u in members:
            r = root[u]
            if r not in local:
                c = Cluster(len(clusters), j)
                clusters.append(c)
                local[r] = c.cid
            cluster_of[u] = local[r]
        assert all(v in mset for u in members for v in G.succ[u])

    for u in G.topo:
        clusters[cluster_of[u]].ops.append(u)
    for c in clusters:
        _summarize(G, c, cluster_of)
    for c in clusters:
        for u in c.ops:
            for tid in G.out_tensors[u]:
                for v in G.tensor_consumers[tid]:
                    d = cluster_of[v]
                    if d != c.cid:
                        c.succs.add(d)
                        lst = clusters[d].preds.setdefault(c.cid, [])
                        if tid not in lst:
                            lst.append(tid)
    return clusters, cluster_of


def cluster_parallelism(G, clusters):
    """簇图并行度：总计算量 / 簇图关键路径（簇按 M+V 计）。"""
    finish = {}
    for cid in cluster_topo(clusters):
        c = clusters[cid]
        finish[cid] = c.M + c.V + max((finish[p] for p in c.preds), default=0)
    return G.total_cycles / max(max(finish.values()), 1)


def single_cluster(G):
    """整图作为一个簇，对应单子图方案。"""
    c = Cluster(0, 0)
    c.ops = list(G.topo)
    _summarize(G, c, {u: 0 for u in G.compute})
    return [c]


def dfs_order(G, members, inside):
    """与官方 step1 相同规则的多源逆向 DFS：最深的出口先展开，前驱中深度大的先访问。"""
    key = lambda v: (G.depth[v], -v)
    sinks = [u for u in members if not any(inside(v) for v in G.succ[u])]
    stack = sorted(sinks, key=key)
    visited, seq = set(), []
    while stack:
        u = stack[-1]
        if u in visited:
            stack.pop()
            continue
        todo = [p for p in G.pred[u] if inside(p) and p not in visited]
        if todo:
            stack.extend(sorted(todo, key=key))
        else:
            visited.add(u)
            seq.append(u)
            stack.pop()
    return seq


def _summarize(G, c, cluster_of):
    """簇的 M/V 工作量、关键路径、独占一核时的顺序发射时长，以及核内执行计划。"""
    cid = c.cid
    seq = dfs_order(G, c.ops, lambda v: cluster_of[v] == cid)
    pos = {u: i for i, u in enumerate(seq)}
    c.M = c.V = 0
    finish = []
    tM = tV = 0.0
    for u in seq:
        cyc = G.cycles[u]
        intra = tuple(pos[p] for p in G.pred[u] if cluster_of[p] == cid)
        ext = tuple((tid, size, -1 if p < 0 else cluster_of[p])
                    for tid, size, p in G.op_inputs[u] if p < 0 or cluster_of[p] != cid)
        c.plan.append((G.is_m[u], cyc, intra, ext))
        r = max((finish[j] for j in intra), default=0.0)
        if G.is_m[u]:
            c.M += cyc
            tM = max(r, tM) + cyc
            finish.append(tM)
        else:
            c.V += cyc
            tV = max(r, tV) + cyc
            finish.append(tV)
        for tid, size in G.inputs[u].items():
            c.inputs[tid] = size
        c.out_bytes += G.out_bytes[u]
    c.solo = max(finish)
    cp = {}
    for u in c.ops:
        cp[u] = G.cycles[u] + max((cp[p] for p in G.pred[u] if cluster_of[p] == cid), default=0)
    c.cp = max(cp.values())


def cluster_topo(clusters):
    indeg = [len(c.preds) for c in clusters]
    order = [c.cid for c in clusters if indeg[c.cid] == 0]
    k = 0
    while k < len(order):
        u = order[k]
        k += 1
        for v in sorted(clusters[u].succs):
            indeg[v] -= 1
            if indeg[v] == 0:
                order.append(v)
    if len(order) != len(clusters):
        raise ValueError('簇图存在环')
    return order
