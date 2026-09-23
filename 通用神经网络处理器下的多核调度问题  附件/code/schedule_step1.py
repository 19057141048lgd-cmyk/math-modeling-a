"""
schedule_step1.py — 步骤 1: 多源 reverse DFS 拓扑排序

对应文档: docs/核内调度算法.md § 1

输入接口:
    step1_schedule(graph_json):
        graph_json: dict, 包含 'ops' (含 'id', 'op', 'pipe', 'cycles') 和 'edges' (含 'source', 'target')
                    可选 'tensors' (二部图需要), 若无 tensors 则假定 edges 已是 op-op 直接边

    step1_from_adj(pred_map, succ_map, V, ops=None):
        pred_map: Dict[v_id -> Set[pred_id]]
        succ_map: Dict[v_id -> Set[succ_id]]
        V: List[v_id] 节点集合
        ops: List[{id, op, ...}] 可选, 提供 op 字段用于判定 is_copy_in / is_copy_out

输出:
    seq: List[v_id] 调度序列 (访问顺序 = 物理执行顺序)

不变量 (由 1.4 命题保证):
    1. 完整覆盖: len(seq) == len(V), seq 是 V 的双射
    2. 拓扑正确: ∀ (u,v) ∈ E: pos(u) < pos(v)
    3. 栈底不变量: seq[-1] = argmin_{v ∈ V_out} start_key(v) (V_out = {v | succ(v) = ∅})
    4. 确定性: 同输入必得同输出
"""

from collections import deque



# ==================== Step 1.0: 深度预计算 ====================

def precompute_depth(pred_map, succ_map, V):
    """
    Step 1.0: 节点深度预计算

    depth(v) = 0                                 if pred(v) = ∅
             = 1 + max_{p ∈ pred(v)} depth(p)   otherwise

    复杂度: O(|V| + |E|)

    输入:
        pred_map: dict[v_id -> set of pred_id]
        succ_map: dict[v_id -> set of succ_id]
        V: list of v_id
    输出:
        depth: dict[v_id -> int]
    约束：邻接表是已验证原图收缩得到的DAG；step1_from_adj原样传入，
    不在此处重复判环。调用方不能将未经入口验证的任意邻接表传入。
    """
    depth = {}
    queue = deque()
    for v in V:
        if len(pred_map[v]) == 0:
            depth[v] = 0
            queue.append(v)
    while queue:
        u = queue.popleft()
        for w in succ_map[u]:
            new_depth = depth[u] + 1
            if w not in depth or new_depth > depth[w]:
                depth[w] = new_depth
                queue.append(w)
    return depth


# ==================== 0.5 节: 二部图 → op-op 邻接表 ====================

def bipartite_to_op_adj(graph_json):
    """
    将附件计算图的 op-tensor 二部图转换为 op-op 邻接表.

    输入: graph_json 含 'ops'、'tensors'、'edges'，两类节点ID唯一且非负
    输出: (pred_map, succ_map, V) 三元组
    约束：图的ID唯一、端点存在且无环；选手入口 validate_graph 已校验。
    转换保留所有tensor生产者和直接op边，因此邻接表与原依赖等价。
    """
    ops = graph_json['ops']
    V = [o['id'] for o in ops]
    op_id_set = set(V)

    # tensor → producer op
    tensor_producer = {}
    for e in graph_json['edges']:
        if e['source'] in op_id_set and e['target'] not in op_id_set:
            tensor_producer.setdefault(e['target'], set()).add(e['source'])

    pred_map = {v: set() for v in V}
    succ_map = {v: set() for v in V}
    for e in graph_json['edges']:
        if e['source'] in op_id_set and e['target'] in op_id_set:
            pred_map[e['target']].add(e['source'])
            succ_map[e['source']].add(e['target'])
        if e['source'] not in op_id_set and e['target'] in op_id_set:
            for producer in tensor_producer.get(e['source'], ()):
                pred_map[e['target']].add(producer)
                succ_map[producer].add(e['target'])

    return pred_map, succ_map, V


# ==================== 步骤 1 主算法 ====================

def step1_from_adj(pred_map, succ_map, V, ops=None):
    """
    Step 1: 多源 reverse DFS 拓扑排序 (核心算法)

    对应文档 1.3 节 Step 1.0 / 1.A / 1.B / 1.C

    排序 key:
        key(p)       = (¬is_copy_in(p),  depth(p), -id(p))   # 主循环: 多前驱展开
        start_key(p) = (¬is_copy_out(p), depth(p), -id(p))   # 起始: 多源压栈

    升序压栈 → LIFO 弹栈 → key 大的先访问 (物理时间上先执行)

    输入:
        pred_map: dict[v_id -> set of pred_id]
        succ_map: dict[v_id -> set of succ_id]
        V: list of v_id
        ops: optional, list of {id, op, ...}  # 提供 op 字段用于判定 is_copy_in / is_copy_out
    输出:
        seq: list of v_id 调度序列
    """
    # 输入约束：V唯一，前驱/后继互为镜像且无环。入口已验证原图，
    # bipartite_to_op_adj成对添加依赖且收缩tensor不产生新环，不重复检查。
    if not V:
        return []
    # ---------- Step 1.0 深度预计算 ----------
    depth = precompute_depth(pred_map, succ_map, V)

    # 判定函数
    is_copy_in = {}
    is_copy_out = {}
    if ops is not None:
        is_copy_in = {o['id']: o['op'] == 'COPY_IN' for o in ops}
        is_copy_out = {o['id']: o['op'] == 'COPY_OUT' for o in ops}
    else:
        is_copy_in = {v: False for v in V}
        is_copy_out = {v: False for v in V}

    def key(v):
        return (not is_copy_in[v], depth[v], -v)

    def start_key(v):
        return (not is_copy_out[v], depth[v], -v)

    # ---------- Step 1.A 多源起始 ----------
    out_nodes = [v for v in V if len(succ_map[v]) == 0]
    if not out_nodes:
        # 退化: 无 out 节点 (环或孤立图) → fallback 到 id 最小
        out_nodes = [min(V)]
    sorted_starts = sorted(out_nodes, key=start_key)
    # 升序压栈: key 小的先入栈 → 栈底; key 大的后入栈 → 栈顶
    stack = list(sorted_starts)

    visited = set()
    seq = []

    # ---------- Step 1.B 主循环 ----------
    while stack or len(visited) < len(V):
        if not stack:
            # 栈空 (多源全清空后) 处理剩余
            remaining = [v for v in V if v not in visited]
            if not remaining:
                break
            for v in sorted(remaining, key=key):
                stack.append(v)
            continue

        u = stack[-1]
        if u in visited:
            stack.pop()
            continue

        unvisited_preds = [p for p in pred_map[u] if p not in visited]
        if unvisited_preds:
            for p in sorted(unvisited_preds, key=key):
                stack.append(p)
        else:
            visited.add(u)
            seq.append(u)
            stack.pop()

    return seq


def step1_schedule(graph_json):
    """
    Step 1 入口: 接受附件计算图格式的 graph_json, 返回调度序列.

    内部: 0.5 预处理 → 步骤 1 主算法
    输入约束：图格式正确且为DAG。保证来源：选手入口已校验原图；Task
    构造保留局部依赖，将跨核边替换为源/汇COPY，不引入局部环。
    本函数为内部算法接口，不重复扫描输入。
    """
    pred_map, succ_map, V = bipartite_to_op_adj(graph_json)
    ops = graph_json['ops']
    return step1_from_adj(pred_map, succ_map, V, ops)


# ==================== CLI ====================


def _check_topo(graph_json, seq):
    """拓扑正确性检查"""
    op_id_set = {o['id'] for o in graph_json['ops']}
    pred_map, succ_map, _ = bipartite_to_op_adj(graph_json)
    pos = {v: i for i, v in enumerate(seq)}
    for u in seq:
        for v in succ_map[u]:
            if pos[u] > pos[v]:
                return False
    return True
