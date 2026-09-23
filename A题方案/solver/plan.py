"""调度结果 -> 官方 plan JSON（node_to_subgraph + core_schedules），以及合法性自检。"""
from collections import defaultdict


def plan_scene_a(clusters, S):
    """问题 1：调度器显式构造的每个 Task 即一个子图，核内按创建顺序执行。"""
    node_to_subgraph = {}
    core_schedules = [[] for _ in range(S.n_cores)]
    sg = 0
    for task in S.tasks:
        if not task['clusters']:
            continue
        for cid in task['clusters']:
            for u in clusters[cid].ops:
                node_to_subgraph[str(u)] = sg
        core_schedules[task['core']].append(sg)
        sg += 1
    return {'node_to_subgraph': node_to_subgraph, 'core_schedules': core_schedules}


def plan_scene_b(clusters, S, chunk_cycles):
    """问题 2/3：每个核上的簇按调度提交顺序切段，每段一个子图，用来固定核内执行顺序。

    官方场景 B 把同核子图按 core_schedules 顺序合并，段内仍按 step1 的 DFS 序排列，
    且跨核 COPY_OUT 会落在源子图末尾、跨核 COPY_IN 落在最早消费子图内。因此：
      * 有跨核前驱的簇另起一段（COPY_IN 位置与模型一致，避免 MTE2 队头阻塞）；
      * 有跨核后继的簇之后立即断段（COPY_OUT 紧跟生产者发出）；
      * 段内累计计算量超过 chunk_cycles 时断段。
    提交顺序 idx 是簇图的拓扑序，段按段首 idx 排序，子图间的边都从小 key 指向大 key，
    商图必然无环且与核内顺序一致。
    """
    order_idx = {cid: i for i, cid in enumerate(S.seq)}
    per_core = defaultdict(list)
    for cid in S.seq:
        per_core[S.core[cid]].append(cid)
    node_to_subgraph = {}
    core_schedules = [[] for _ in range(S.n_cores)]
    chunks = []  # (key, core, [cids])
    for k, cids in per_core.items():
        cur, head, weight = [], None, 0.0
        for x in cids:
            c = clusters[x]
            remote_in = any(S.core[p] != k for p in c.preds)
            if cur and (remote_in or weight + c.M + c.V > chunk_cycles):
                chunks.append((head, k, cur))
                cur, weight = [], 0.0
            if not cur:
                head = order_idx[x]
            cur.append(x)
            weight += c.M + c.V
            if any(S.core[s] != k for s in c.succs):
                chunks.append((head, k, cur))
                cur, weight = [], 0.0
        if cur:
            chunks.append((head, k, cur))
    chunks.sort(key=lambda t: t[0])
    for sg, (_, k, cids) in enumerate(chunks):
        for cid in cids:
            for u in clusters[cid].ops:
                node_to_subgraph[str(u)] = sg
        core_schedules[k].append(sg)
    return {'node_to_subgraph': node_to_subgraph, 'core_schedules': core_schedules}


def validate_plan(G, plan):
    """与官方校验同口径的快速自检：覆盖全部计算算子、每个子图恰好出现一次、商图无环、核内顺序无死锁。"""
    mapping = plan['node_to_subgraph']
    missing = [u for u in G.compute if str(u) not in mapping]
    if missing:
        raise ValueError('有 %d 个计算算子未分配子图' % len(missing))
    seen = [sg for order in plan['core_schedules'] for sg in order]
    if len(seen) != len(set(seen)) or set(seen) != set(mapping.values()):
        raise ValueError('core_schedules 中子图必须恰好出现一次')
    edges = defaultdict(set)
    for u in G.compute:
        a = mapping[str(u)]
        for v in G.succ[u]:
            b = mapping[str(v)]
            if a != b:
                edges[a].add(b)
    for order in plan['core_schedules']:
        for a, b in zip(order, order[1:]):
            edges[a].add(b)
    indeg = defaultdict(int)
    for a, bs in edges.items():
        for b in bs:
            indeg[b] += 1
    stack = [sg for sg in set(seen) if indeg[sg] == 0]
    visited = 0
    while stack:
        a = stack.pop()
        visited += 1
        for b in edges.get(a, ()):
            indeg[b] -= 1
            if indeg[b] == 0:
                stack.append(b)
    if visited != len(set(seen)):
        raise ValueError('子图商图与核内顺序构成环（会死锁）')
    return len(set(seen))
