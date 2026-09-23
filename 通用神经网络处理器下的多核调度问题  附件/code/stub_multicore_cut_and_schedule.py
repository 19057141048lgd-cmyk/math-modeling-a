"""生成随机、合法、无优化目标的多核切图与子图调度方案。

方案 JSON 只包含 node_to_subgraph 和 core_schedules；子图成员、依赖、
核归属和核数均由原图与这两个字段推导。
"""

import random
from collections import defaultdict
from evaluation_validation import validate_graph, require_integer

DEFAULT_NUM_CORES = 4
DEFAULT_SEED = 0
DEFAULT_MIN_SUBGRAPH_SIZE = 50
DEFAULT_MAX_SUBGRAPH_SIZE = 100
EXCLUDED_COPY_TYPES = {'COPY_IN', 'COPY_OUT'}

class MulticoreCutError(RuntimeError):
    """输入图有环，或参数/生成结果不合法。"""

def _build_op_adjacency(graph_json):
    """把 tensor 中转边和直接 op-op 边统一转换成 op DAG。"""
    op_ids = {op['id'] for op in graph_json.get('ops', [])}
    preds = {op_id: set() for op_id in op_ids}
    succs = {op_id: set() for op_id in op_ids}
    producers = defaultdict(set)
    consumers = defaultdict(set)
    for edge in graph_json.get('edges', []):
        src, dst = edge['source'], edge['target']
        src_is_op, dst_is_op = src in op_ids, dst in op_ids
        if src_is_op and dst_is_op and src != dst:
            succs[src].add(dst)
            preds[dst].add(src)
        elif src_is_op and not dst_is_op:
            producers[dst].add(src)
        elif not src_is_op and dst_is_op:
            consumers[src].add(dst)
    for tensor_id, producer_ids in producers.items():
        for src in producer_ids:
            for dst in consumers.get(tensor_id, ()):
                if src != dst:
                    succs[src].add(dst)
                    preds[dst].add(src)
    return preds, succs

def _random_topological_order(node_ids, preds, succs, rng):
    """Kahn 算法，每次从 ready 集合随机选择一个节点。"""
    node_set = set(node_ids)
    indegree = {
        node_id: sum(pred in node_set for pred in preds[node_id])
        for node_id in node_ids
    }
    ready = sorted(node_id for node_id in node_ids if indegree[node_id] == 0)
    order = []
    while ready:
        node_id = ready.pop(rng.randrange(len(ready)))
        order.append(node_id)
        for succ_id in sorted(succs[node_id]):
            if succ_id not in node_set:
                continue
            indegree[succ_id] -= 1
            if indegree[succ_id] == 0:
                ready.append(succ_id)
                ready.sort()
    if len(order) != len(node_ids):
        unresolved = sorted(node_set - set(order))
        raise MulticoreCutError(
            'input graph contains a cycle; unresolved op ids={}'.format(unresolved))
    return order

def _contract_excluded_copy_nodes(eligible_ids, succs):
    """跳过 COPY_IN/COPY_OUT，保留其两侧最近 eligible op 的依赖。"""
    eligible = set(eligible_ids)
    contracted_succs = {node_id: set() for node_id in eligible_ids}
    contracted_preds = {node_id: set() for node_id in eligible_ids}
    for src_id in eligible_ids:
        stack = list(sorted(succs[src_id], reverse=True))
        visited_excluded = set()
        while stack:
            dst_id = stack.pop()
            if dst_id in eligible:
                if dst_id != src_id:
                    contracted_succs[src_id].add(dst_id)
                    contracted_preds[dst_id].add(src_id)
                continue
            if dst_id in visited_excluded:
                continue
            visited_excluded.add(dst_id)
            stack.extend(sorted(succs.get(dst_id, ()), reverse=True))
    return contracted_preds, contracted_succs

def _random_partition_sizes(total, min_size, max_size, rng):
    """随机生成不超过 max_size 的块；最后一块允许少于 min_size。"""
    if total == 0:
        return []
    sizes = []
    remaining = total
    while remaining > max_size:
        upper = min(max_size, remaining - min_size)
        size = rng.randint(min_size, upper)
        sizes.append(size)
        remaining -= size
    sizes.append(remaining)
    return sizes

def validate_multicore_plan(graph_json, plan):
    """方案基础入口：校验原图、节点覆盖、子图DAG和同核直接依赖顺序。

    不保证各题的执行可行性：题目1还需合并Task顺序，题目2/3还需合并
    Step3的Pipe/内存依赖及跨核COPY。这些新约束由对应评估入口检查。
    """
    derive_multicore_plan(graph_json, plan)
    return True

def derive_multicore_plan(graph_json, plan):
    """从最小输入方案推导消费者所需的子图与依赖视图。"""
    validate_graph(graph_json)
    required_fields = {'node_to_subgraph', 'core_schedules'}
    if not isinstance(plan, dict) or set(plan) != required_fields:
        raise MulticoreCutError(
            'multicore plan must contain exactly {}'.format(
                sorted(required_fields)))
    eligible = {
        op['id'] for op in graph_json.get('ops', [])
        if op.get('op') not in EXCLUDED_COPY_TYPES
    }
    if not isinstance(plan['node_to_subgraph'], dict):
        raise MulticoreCutError('node_to_subgraph must be an object')
    if any(not isinstance(subgraph_id, int) or isinstance(subgraph_id, bool)
           for subgraph_id in plan['node_to_subgraph'].values()):
        raise MulticoreCutError(
            'node_to_subgraph values must be integer subgraph ids')
    try:
        if any(type(node_id) is not int and not (
                isinstance(node_id, str) and node_id.isascii() and node_id.isdecimal())
                for node_id in plan['node_to_subgraph']):
            raise ValueError('non-integer key')
        mapping = {
            int(node_id): subgraph_id
            for node_id, subgraph_id in plan['node_to_subgraph'].items()
        }
    except (TypeError, ValueError):
        raise MulticoreCutError(
            'node_to_subgraph keys must be integer op ids')
    if len(mapping) != len(plan['node_to_subgraph']):
        raise MulticoreCutError(
            'node_to_subgraph contains duplicate integer op ids')
    if set(mapping) != eligible:
        missing = sorted(eligible - set(mapping))
        extra = sorted(set(mapping) - eligible)
        raise MulticoreCutError(
            'node_to_subgraph must exactly cover non-COPY ops; '
            'missing={} extra={}'.format(missing[:20], extra[:20]))
    if any(subgraph_id < 0 for subgraph_id in mapping.values()):
        raise MulticoreCutError('subgraph ids must be non-negative')

    schedules = plan['core_schedules']
    if not isinstance(schedules, list) or not schedules:
        raise MulticoreCutError('core_schedules must be a non-empty list')
    if any(not isinstance(order, list) for order in schedules):
        raise MulticoreCutError(
            'each core_schedules entry must be a subgraph id list')
    if any(not isinstance(subgraph_id, int) or isinstance(subgraph_id, bool)
           for order in schedules for subgraph_id in order):
        raise MulticoreCutError('core schedule values must be integers')
    core_orders = {
        core_id: list(order) for core_id, order in enumerate(schedules)}

    subgraph_ids = set(mapping.values())
    scheduled = [
        subgraph_id
        for core_id in range(len(schedules))
        for subgraph_id in core_orders[core_id]
    ]
    if len(scheduled) != len(set(scheduled)) or set(scheduled) != subgraph_ids:
        raise MulticoreCutError('core schedules must cover every subgraph exactly once')

    nodes_by_subgraph = {subgraph_id: [] for subgraph_id in subgraph_ids}
    for node_id, subgraph_id in mapping.items():
        nodes_by_subgraph[subgraph_id].append(node_id)
    for node_ids in nodes_by_subgraph.values():
        node_ids.sort()
    core_by_subgraph = {
        subgraph_id: core_id
        for core_id, order in core_orders.items()
        for subgraph_id in order
    }

    _, full_succs = _build_op_adjacency(graph_json)
    _, contracted_succs = _contract_excluded_copy_nodes(
        sorted(eligible), full_succs)
    dependency_pairs = sorted({
        (mapping[src_id], mapping[dst_id])
        for src_id in eligible
        for dst_id in contracted_succs[src_id]
        if mapping[src_id] != mapping[dst_id]
    })

    subgraph_preds = {subgraph_id: set() for subgraph_id in subgraph_ids}
    subgraph_succs = {subgraph_id: set() for subgraph_id in subgraph_ids}
    for source, target in dependency_pairs:
        subgraph_succs[source].add(target)
        subgraph_preds[target].add(source)
    indegree = {
        subgraph_id: len(subgraph_preds[subgraph_id])
        for subgraph_id in subgraph_ids
    }
    ready = sorted(
        subgraph_id for subgraph_id, degree in indegree.items()
        if degree == 0)
    visited = []
    while ready:
        subgraph_id = ready.pop(0)
        visited.append(subgraph_id)
        for successor in sorted(subgraph_succs[subgraph_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(visited) != len(subgraph_ids):
        raise MulticoreCutError('contracted subgraph graph contains a cycle')

    for core_id, order in core_orders.items():
        position = {
            subgraph_id: index for index, subgraph_id in enumerate(order)}
        for source, target in dependency_pairs:
            if (core_by_subgraph[source] == core_id
                    and core_by_subgraph[target] == core_id
                    and position[source] >= position[target]):
                raise MulticoreCutError(
                    'dependency order violation on core {}: {} -> {}'.format(
                        core_id, source, target))

    return {
        'mapping': mapping,
        'num_cores': len(schedules),
        'subgraph_ids': sorted(subgraph_ids),
        'nodes_by_subgraph': nodes_by_subgraph,
        'core_orders': core_orders,
        'core_by_subgraph': core_by_subgraph,
        'dependency_pairs': dependency_pairs,
        'subgraph_preds': subgraph_preds,
        'subgraph_succs': subgraph_succs,
    }

def generate_multicore_plan(graph_json, num_cores=DEFAULT_NUM_CORES,
                            seed=DEFAULT_SEED,
                            min_subgraph_size=DEFAULT_MIN_SUBGRAPH_SIZE,
                            max_subgraph_size=DEFAULT_MAX_SUBGRAPH_SIZE):
    """生成由 seed 确定、可复现的随机切图与调度方案。"""
    validate_graph(graph_json)
    require_integer(num_cores, 'num_cores', 1)
    require_integer(min_subgraph_size, 'min_subgraph_size', 1)
    require_integer(max_subgraph_size, 'max_subgraph_size', 1)
    if num_cores <= 0:
        raise MulticoreCutError('num_cores must be positive, got {}'.format(num_cores))
    if min_subgraph_size <= 0 or max_subgraph_size < min_subgraph_size:
        raise MulticoreCutError(
            'invalid subgraph size range: {}..{}'.format(
                min_subgraph_size, max_subgraph_size))

    rng = random.Random(seed)
    op_by_id = {op['id']: op for op in graph_json.get('ops', [])}
    preds, succs = _build_op_adjacency(graph_json)
    # 原图已在入口验证；收缩COPY保留可达性，不重复扫描原图。

    eligible_ids = sorted(
        op_id for op_id, op in op_by_id.items()
        if op.get('op') not in EXCLUDED_COPY_TYPES)
    contracted_preds, contracted_succs = _contract_excluded_copy_nodes(
        eligible_ids, succs)
    topo_order = _random_topological_order(
        eligible_ids, contracted_preds, contracted_succs, rng)
    sizes = _random_partition_sizes(
        len(topo_order), min_subgraph_size, max_subgraph_size, rng)

    node_to_subgraph = {}
    cursor = 0
    for subgraph_id, size in enumerate(sizes):
        node_ids = topo_order[cursor:cursor + size]
        cursor += size
        for node_id in node_ids:
            node_to_subgraph[node_id] = subgraph_id

    core_schedules = [[] for _ in range(num_cores)]
    for subgraph_id in range(len(sizes)):
        core_id = rng.randrange(num_cores)
        core_schedules[core_id].append(subgraph_id)

    plan = {
        'node_to_subgraph': node_to_subgraph,
        'core_schedules': core_schedules,
    }
    # 连续全局拓扑区间构成凸分区，按递增子图ID投影到每核，覆盖唯一且
    # 数据边和每核顺序均沿全局拓扑方向，不需要再次验证生成结果。
    return plan

if __name__ == '__main__':
    from contest_io import run_multicore_stub_cli
    raise SystemExit(run_multicore_stub_cli())
