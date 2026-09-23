import math
from collections import deque
from pathlib import Path


class EvaluationValidationError(ValueError):
    """An input or fixed execution order cannot be evaluated."""

PIPES = ('PIPE_MTE2', 'PIPE_MTE3', 'PIPE_M', 'PIPE_V')

def require_integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise EvaluationValidationError(f'{name} must be an integer >= {minimum}; got {value!r}')

def require_number(value, name, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise EvaluationValidationError(f'{name} must be finite and {"> 0" if positive else ">= 0"}; got {value!r}')

def validate_capacity(capacity):
    if not isinstance(capacity, dict) or set(capacity) != {'L1', 'UB'}:
        raise EvaluationValidationError('capacity must contain exactly L1 and UB')
    for name, size in capacity.items():
        require_integer(size, f'capacity.{name}')

CONFIG_SECTIONS = ('capacity', 'bandwidth', 'multicore_scene_a',
                   'multicore_scene_b', 'problem_3')

def _iter_settings(path):
    """逐行解析 config.txt，产出 (行号, 段名, 键, 值)。

    段名带方括号时必须落在 CONFIG_SECTIONS 内：这样 `[bandwidht]` 之类的
    拼写错误，以及已经废弃的段（例如早期版本的 `[pipe_capacity]`）会立刻
    报错，而不是被静默忽略、让选手以为配置生效了。
    """
    active = None
    for line_no, line in enumerate(
            Path(path).read_text(encoding='utf-8').splitlines(), 1):
        text = line.split('#', 1)[0].strip()
        if not text:
            continue
        if text.startswith('[') and text.endswith(']'):
            name = text[1:-1].strip().lower()
            if name not in CONFIG_SECTIONS:
                raise EvaluationValidationError(
                    f'{path}:{line_no}: unknown section [{name}]')
            active = name
            continue
        parts = text.split()
        if len(parts) != 2 or active is None:
            raise EvaluationValidationError(
                f'{path}:{line_no}: setting must be "<key> <integer>" '
                f'inside one of {", ".join(CONFIG_SECTIONS)}: {text}')
        yield line_no, active, parts[0], parts[1]

def read_integer_settings(path, section, keys):
    """Missing default config is allowed; invalid settings never silently default."""
    result = {}
    if not path or not Path(path).exists():
        return result
    for line_no, name, key, value in _iter_settings(path):
        if name != section:
            continue
        if key not in keys or key in result:
            raise EvaluationValidationError(f'{path}:{line_no}: invalid/duplicate [{section}] setting: {key}')
        try:
            result[key] = int(value)
        except ValueError:
            raise EvaluationValidationError(f'{path}:{line_no}: {key} must be an integer') from None
    return result

def read_required_settings(path, section, keys):
    """读取必需配置段；段缺失或键缺失都直接报错，不使用代码默认值。"""
    result = read_integer_settings(path, section, keys)
    missing = sorted(set(keys) - set(result))
    if missing:
        raise EvaluationValidationError(
            f'{path}: [{section}] must define {", ".join(missing)}')
    for key, value in result.items():
        require_integer(value, key)
    return result

def read_capacity_config(path):
    """读取 [capacity]；该项无代码默认值，必须由 config.txt 给出 L1/UB。"""
    return read_required_settings(path, 'capacity', ('L1', 'UB'))

def read_bandwidth_config(path):
    """读取 [bandwidth]；带宽必须为正整数，且不允许回退到代码默认值。"""
    value = read_required_settings(path, 'bandwidth', ('bandwidth',))['bandwidth']
    require_number(value, 'bandwidth', positive=True)
    return value

def read_evaluation_config(path):
    """评估脚本的统一配置入口：容量与带宽全部来自 config.txt。

    显式给出或默认推导出的配置文件都不存在时直接报错，不再回退到代码里的
    默认数值，避免用错误的固定参数评出成绩。
    """
    if not path or not Path(path).exists():
        raise EvaluationValidationError(f'configuration file not found: {path}')
    return {
        'capacity': read_capacity_config(path),
        'bandwidth': read_bandwidth_config(path),
    }

def validate_parameters(bandwidth, capacity, max_iter, **delays):
    require_number(bandwidth, 'bandwidth', positive=True)
    validate_capacity(capacity)
    require_integer(max_iter, 'max_iter', 1)
    for name, delay in delays.items():
        require_number(delay, name)

def check_acyclic(nodes, edges, context):
    """Linear Kahn check; iterative DFS on failure returns labelled ring edges.

    Nodes may be ids or (core, op) pairs. Edges are (source, target, reason).
    Avoid recursion limits on the large official graphs.
    """
    adjacency = {node: {} for node in nodes}
    indegree = dict.fromkeys(adjacency, 0)
    for source, target, reason in edges:
        if source not in adjacency or target not in adjacency:
            raise EvaluationValidationError(f'{context}: unknown edge endpoint {source!r} -> {target!r}')
        if target not in adjacency[source]:
            adjacency[source][target] = reason
            indegree[target] += 1
    ready = deque(node for node, count in indegree.items() if count == 0)
    count = 0
    while ready:
        node = ready.popleft()
        count += 1
        for nxt in adjacency[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    if count == len(adjacency):
        return
    colors = {}
    parent = {}
    for root in adjacency:
        if not indegree[root] or root in colors:
            continue
        colors[root] = 1
        stack = [(root, iter(adjacency[root]))]
        while stack:
            node, children = stack[-1]
            nxt = next(children, None)
            if nxt is None:
                colors[node] = 2
                stack.pop()
                continue
            if colors.get(nxt) == 1:
                ring = [(node, nxt, adjacency[node][nxt])]
                cursor = node
                while cursor != nxt:
                    prev = parent[cursor]
                    ring.append((prev, cursor, adjacency[prev][cursor]))
                    cursor = prev
                ring.reverse()
                text = '; '.join(f'{a!r} -[{why}]-> {b!r}' for a,b,why in ring[:16])
                if len(ring) > 16:
                    text += f'; ... ({len(ring)} edges total)'
                error = EvaluationValidationError(f'{context}: dependency cycle; {text}')
                error.cycle = ring
                raise error
            if nxt not in colors:
                colors[nxt] = 1
                parent[nxt] = node
                stack.append((nxt, iter(adjacency[nxt])))

def validate_graph(graph):
    if not isinstance(graph, dict):
        raise EvaluationValidationError('graph must be an object')
    for field in ('ops', 'tensors', 'edges'):
        if not isinstance(graph.get(field), list):
            raise EvaluationValidationError(f'graph.{field} must be a list')
    ids, ops = set(), set()
    for kind, entries in (('op', graph['ops']), ('tensor', graph.get('tensors', []))):
        for item in entries:
            if not isinstance(item, dict):
                raise EvaluationValidationError(f'{kind} must be an object')
            ident = item.get('id')
            require_integer(ident, f'{kind}.id')
            if ident in ids:
                raise EvaluationValidationError(f'duplicate op/tensor id: {ident}')
            ids.add(ident)
            if kind == 'tensor':
                require_integer(item.get('size'), f'tensor {ident} size')
                if item.get('pos') not in ('DDR', 'L1', 'UB'):
                    raise EvaluationValidationError(f'tensor {ident}: unsupported memory pos {item.get("pos")!r}')
            else:
                ops.add(ident)
                # op 类型名不做白名单校验，Pipe 直接取节点自带的 pipe 字段。
                if item.get('pipe') not in PIPES:
                    raise EvaluationValidationError(
                        f'op {ident}: pipe must be one of {", ".join(PIPES)}; '
                        f'got {item.get("pipe")!r}')
                require_integer(item.get('cycles'), f'op {ident} cycles')
    edges = []
    seen = set()
    for edge in graph['edges']:
        if not isinstance(edge, dict):
            raise EvaluationValidationError('edge must be an object')
        a, b = edge.get('source'), edge.get('target')
        require_integer(a, 'edge.source')
        require_integer(b, 'edge.target')
        if a not in ids or b not in ids:
            raise EvaluationValidationError(f'edge references unknown id: {a} -> {b}')
        if a not in ops and b not in ops:
            raise EvaluationValidationError(f'tensor-to-tensor edge is unsupported: {a} -> {b}')
        if (a,b) in seen:
            raise EvaluationValidationError(f'duplicate edge: {a} -> {b}')
        seen.add((a,b))
        edges.append((a,b,'graph'))
    check_acyclic(sorted(ids), edges, 'input graph')

def validate_task_order(view):
    # 无相邻Task顺序边时，没有新增约束；子图DAG已由方案入口保证。
    if not any(len(order) > 1 for order in view['core_orders'].values()):
        return
    edges = [(a,b,'subgraph dependency') for a,b in view['dependency_pairs']]
    for core, order in view['core_orders'].items():
        edges += [(a,b,f'core {core} task order') for a,b in zip(order,order[1:])]
    check_acyclic(view['subgraph_ids'], edges, 'task schedule')

def validate_execution(tasks, cross_links):
    """Check actual local dependencies + FIFO + external COPY, not serial tasks."""
    # Step3已保证本地执行图和Pipe覆盖合法；此处只检查跨核组合新增的环。
    if not cross_links:
        return
    nodes, edges = [], []
    for core, task in tasks.items():
        ids = set(task['op_by_id'])
        nodes.extend((core,op) for op in sorted(ids))
        for op, preds in task['op_preds'].items():
            edges.extend(((core,pred),(core,op),'local data/memory dependency') for pred in sorted(preds))
        for pipe, order in task['pipe_ops'].items():
            edges.extend(((core,a),(core,b),f'{pipe} FIFO') for a,b in zip(order,order[1:]))
    for link in cross_links:
        source = (link['source_core'],link['source_copy_out_id'])
        target = (link['target_core'],link['target_copy_in_id'])
        edges.append((source,target,'cross-core COPY'))
    check_acyclic(nodes, edges, 'global execution (core, op)')
