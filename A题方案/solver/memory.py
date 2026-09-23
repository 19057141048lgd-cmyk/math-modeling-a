"""片上容量（spill）模型。

按附录 C 复现每个 Task 的执行序列（step1 多源逆向 DFS；场景 B/C 再按子图名次稳定排序），
再按 step2 的 Belady 规则模拟 L1/UB 驻留，得到换出写回与换入的字节数。
Task 的构造（边界 COPY 的插入与归属子图）与官方评估器一致，因此估计值可以逐字节对上官方的
spill_added_copy_bytes；淘汰对象用堆维护，复杂度 O(n log n)。
"""
import heapq
import json
from collections import defaultdict
from functools import lru_cache

COMPUTE, COPY_IN, COPY_OUT = 0, 1, 2
NEW_ID_BASE = 1 << 40   # 新增 COPY 节点只与同 Task 内的 COPY 节点比较编号，保持构造顺序即可


class Raw:
    """原始二部图中与 Task 构造、驻留相关的部分。"""

    def __init__(self, path):
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
        self.kind = {o['id']: o['op'] for o in raw['ops']}
        self.pos, self.size = {}, {}
        for t in raw['tensors']:
            self.pos[t['id']] = 'UB' if t['pos'] == 'DDR' else t['pos']
            self.size[t['id']] = t['size']
        self.producers, self.consumers = defaultdict(list), defaultdict(list)
        for e in raw['edges']:
            s, d = e['source'], e['target']
            if s in self.kind and d in self.size:
                self.producers[d].append(s)
            elif s in self.size and d in self.kind:
                self.consumers[s].append(d)
            else:
                raise ValueError('不支持算子到算子的直连边')
        self.touch = defaultdict(list)          # 计算/搬运算子 -> 涉及的张量（按编号升序）
        for t in sorted(self.size):
            for u in self.producers.get(t, ()):
                self.touch[u].append(t)
            for u in self.consumers.get(t, ()):
                self.touch[u].append(t)
        self.graph_output = {t for t, cs in self.consumers.items()
                             if any(self.kind[c] == 'COPY_OUT' for c in cs)}


@lru_cache(maxsize=4)
def raw_view(path):
    return Raw(path)


class Task:
    __slots__ = ('core', 'nodes', 'kind', 'ins', 'outs', 'sub')

    def __init__(self, core):
        self.core = core
        self.nodes = []
        self.kind = {}
        self.ins = defaultdict(list)     # 节点 -> 消费的片上张量
        self.outs = defaultdict(list)    # 节点 -> 产出的片上张量
        self.sub = {}                    # 节点 -> 归属子图（场景 B/C 排序用）

    def add(self, node, kind, sub=None):
        self.nodes.append(node)
        self.kind[node] = kind
        self.sub[node] = sub


def _mapping(plan):
    return {int(k): v for k, v in plan['node_to_subgraph'].items()}


def tasks_scene_a(R, plan):
    """每个子图一个 Task：边界输入各自 COPY_IN；图输出或被其他子图消费的张量各自 COPY_OUT。"""
    mapping = _mapping(plan)
    core_of = {sg: k for k, order in enumerate(plan['core_schedules']) for sg in order}
    by_sg = defaultdict(list)
    for u, sg in mapping.items():
        by_sg[sg].append(u)
    nid = NEW_ID_BASE
    tasks = []
    for sg in sorted(by_sg):
        ops = sorted(by_sg[sg])
        inside = set(ops)
        task = Task(core_of[sg])
        for u in ops:
            task.add(u, COMPUTE)
        touched = sorted({t for u in ops for t in R.touch[u]})
        for t in touched:
            lp = [p for p in R.producers.get(t, ()) if p in inside]
            lc = [c for c in R.consumers.get(t, ()) if c in inside]
            for p in lp:
                task.outs[p].append(t)
            for c in lc:
                task.ins[c].append(t)
            if lc and not lp:
                task.add(nid, COPY_IN)
                task.outs[nid].append(t)
                nid += 1
            if lp:
                ec = [c for c in R.consumers.get(t, ()) if c in mapping]
                if t in R.graph_output or not ec or any(mapping[c] != sg for c in ec):
                    task.add(nid, COPY_OUT)
                    task.ins[nid].append(t)
                    nid += 1
        tasks.append(task)
    return tasks


def tasks_scene_b(R, plan):
    """每核一个 Task：图输入每核读一次，跨核张量每个（源核, 目标核）一对 COPY。"""
    mapping = _mapping(plan)
    orders = plan['core_schedules']
    rank = {sg: i for order in orders for i, sg in enumerate(order)}
    sg_core = {sg: k for k, order in enumerate(orders) for sg in order}
    core_of_op = {}
    tasks = [Task(k) for k in range(len(orders))]
    for u, sg in mapping.items():
        core_of_op[u] = sg_core[sg]
        tasks[sg_core[sg]].add(u, COMPUTE, sg)
    nid = NEW_ID_BASE
    touched = sorted({t for u in mapping for t in R.touch[u]})
    for t in touched:
        ep = [p for p in R.producers.get(t, ()) if p in mapping]
        ec = [c for c in R.consumers.get(t, ()) if c in mapping]
        for p in ep:
            tasks[core_of_op[p]].outs[p].append(t)
        for c in ec:
            tasks[core_of_op[c]].ins[c].append(t)
        pc = sorted({core_of_op[p] for p in ep})
        cc = sorted({core_of_op[c] for c in ec})
        first_sg = {d: min((mapping[c] for c in ec if core_of_op[c] == d), key=rank.get) for d in cc}
        last_sg = {s: max((mapping[p] for p in ep if core_of_op[p] == s), key=rank.get) for s in pc}
        if ec and not ep:
            for d in cc:
                tasks[d].add(nid, COPY_IN, first_sg[d])
                tasks[d].outs[nid].append(t)
                nid += 1
        if ep and (t in R.graph_output or not ec):
            for s in pc:
                tasks[s].add(nid, COPY_OUT, last_sg[s])
                tasks[s].ins[nid].append(t)
                nid += 1
        for s in pc:
            for d in cc:
                if s == d:
                    continue
                tasks[s].add(nid, COPY_OUT, last_sg[s])
                tasks[s].ins[nid].append(t)
                tasks[d].add(nid + 1, COPY_IN, first_sg[d])
                tasks[d].outs[nid + 1].append(t)
                nid += 2
    return [task for task in tasks if task.nodes], rank


def step1_sequence(task):
    """官方 step1：多源逆向 DFS。起始按 (非COPY_OUT, depth, -id) 压栈，前驱按 (非COPY_IN, depth, -id) 压栈。"""
    nodes, kind = task.nodes, task.kind
    producer = defaultdict(list)
    for u in nodes:
        for t in task.outs.get(u, ()):
            producer[t].append(u)
    pred = {u: set() for u in nodes}
    succ = {u: [] for u in nodes}
    for v in nodes:
        for t in task.ins.get(v, ()):
            for u in producer.get(t, ()):
                if u not in pred[v]:
                    pred[v].add(u)
                    succ[u].append(v)
    indeg = {v: len(pred[v]) for v in nodes}
    order = [v for v in nodes if indeg[v] == 0]
    depth = dict.fromkeys(order, 0)
    i = 0
    while i < len(order):
        u = order[i]
        i += 1
        for w in succ[u]:
            if depth[u] + 1 > depth.get(w, -1):
                depth[w] = depth[u] + 1
            indeg[w] -= 1
            if indeg[w] == 0:
                order.append(w)

    def key(v):
        return (kind[v] != COPY_IN, depth[v], -v)

    stack = sorted((v for v in nodes if not succ[v]),
                   key=lambda v: (kind[v] != COPY_OUT, depth[v], -v)) or [min(nodes)]
    visited, seq = set(), []
    while stack or len(visited) < len(nodes):
        if not stack:
            stack.extend(sorted((v for v in nodes if v not in visited), key=key))
            continue
        u = stack[-1]
        if u in visited:
            stack.pop()
            continue
        todo = [p for p in pred[u] if p not in visited]
        if todo:
            stack.extend(sorted(todo, key=key))
        else:
            visited.add(u)
            seq.append(u)
            stack.pop()
    return seq


def step2_spill(task, seq, R, capacity):
    """官方 step2 的驻留模拟：alloc 后、释放末次输入前检查容量，超限时换出下次使用最远者。

    返回 (spill 字节, 是否可行)。首次换出且没有 DDR 副本的张量计写回 + 换入两份流量。
    """
    step = {u: i for i, u in enumerate(seq)}
    uses = defaultdict(set)
    backing = set()
    for u in seq:
        s = step[u]
        for t in task.outs.get(u, ()):
            uses[t].add(s)
            if task.kind[u] == COPY_IN:
                backing.add(t)
        for t in task.ins.get(u, ()):
            uses[t].add(s)
    at = defaultdict(list)
    for t in sorted(uses):
        us = sorted(uses[t])
        uses[t] = us
        for idx, s in enumerate(us):
            at[s].append((t, idx))
    size, pos = R.size, R.pos
    active = {T: {} for T in capacity}       # 张量 -> [下次使用步, 已使用次数, 插入序号]
    heap = {T: [] for T in capacity}
    resid = dict.fromkeys(capacity, 0)
    reload_at = defaultdict(list)
    spills = []
    serial = 0
    for s in range(len(seq)):
        for t, u in reload_at.pop(s, ()):
            T = pos[t]
            if t not in active[T]:
                us = uses[t]
                nu = us[u + 1] if u + 1 < len(us) else None
                serial += 1
                active[T][t] = [nu, u, serial]
                resid[T] += size[t]
                if nu is not None:
                    heapq.heappush(heap[T], (-nu, serial, t))
        current = {T: set() for T in capacity}
        release = []
        for t, idx in at[s]:
            T = pos[t]
            current[T].add(t)
            us = uses[t]
            if idx == 0:
                nu = us[1] if len(us) > 1 else None
                serial += 1
                active[T][t] = [nu, 1, serial]
                resid[T] += size[t]
                if nu is not None:
                    heapq.heappush(heap[T], (-nu, serial, t))
            elif idx < len(us) - 1:
                a = active[T].get(t)
                if a is not None:
                    a[0] = us[idx + 1]
                    a[1] += 1
                    heapq.heappush(heap[T], (-a[0], a[2], t))
            if idx == len(us) - 1:
                release.append((T, t))
        for T, C in capacity.items():
            skipped = []
            while resid[T] > C:
                victim = None
                while heap[T]:
                    neg, ser, t = heapq.heappop(heap[T])
                    a = active[T].get(t)
                    if a is None or a[2] != ser or a[0] != -neg:
                        continue
                    if t in current[T]:
                        skipped.append((neg, ser, t))
                        continue
                    victim = t
                    break
                if victim is None:
                    return sum(spills), False
                nu, ui, _ = active[T].pop(victim)
                resid[T] -= size[victim]
                spills.append(size[victim] * (1 if victim in backing else 2))
                backing.add(victim)
                reload_at[nu].append((victim, ui))
            for item in skipped:
                heapq.heappush(heap[T], item)
        for T, t in release:
            if active[T].pop(t, None) is not None:
                resid[T] -= size[t]
    return sum(spills), True


def estimate_spill(G, plan, scene, hw):
    """返回 (总 spill 字节, {核: spill 字节}, 是否可行)。场景 'A' 按子图成 Task，'B'/'C' 按核成 Task。"""
    R = raw_view(G.path)
    capacity = {'L1': hw.l1, 'UB': hw.ub}
    if scene == 'A':
        tasks, rank = tasks_scene_a(R, plan), None
    else:
        tasks, rank = tasks_scene_b(R, plan)
    per_core = defaultdict(int)
    feasible = True
    for task in tasks:
        seq = step1_sequence(task)
        if rank is not None:
            seq.sort(key=lambda u: rank[task.sub[u]])
        b, ok = step2_spill(task, seq, R, capacity)
        per_core[task.core] += b
        feasible &= ok
    return sum(per_core.values()), dict(per_core), feasible
