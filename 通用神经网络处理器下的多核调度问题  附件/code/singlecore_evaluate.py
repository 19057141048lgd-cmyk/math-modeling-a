"""单核执行评估入口；无需多核切图与调度方案。"""

import argparse
import json
import os

from contest_io import (
    cli_errors, _read_json,
    emit,
    format_data_movement_log,
    format_multicore_result_log,
    format_scene_a_trace_json,
    write_text,
)
from multicore_cut_evaluate_problem_1 import (
    evaluate_scene_a,
    read_scene_a_config,
)
from evaluation_validation import read_evaluation_config
from stub_multicore_cut_and_schedule import EXCLUDED_COPY_TYPES

def build_singlecore_plan(graph_json):
    """将全部非 COPY 操作合并为一个子图并调度到核心 0。"""
    # 完整原图校验由后续evaluate_scene_a入口执行；这里只要求能够枚举ID，
    # 避免非法JSON在到达该入口之前触发KeyError/TypeError。
    try:
        eligible = sorted(
            op['id'] for op in graph_json.get('ops', [])
            if op.get('op') not in EXCLUDED_COPY_TYPES)
    except (AttributeError, KeyError, TypeError):
        raise ValueError('graph.ops must be a list of op objects with integer ids') from None
    return {
        'node_to_subgraph': {
            str(op_id): 0 for op_id in eligible
        },
        'core_schedules': [[0] if eligible else []],
    }

def evaluate_singlecore(graph_json, bandwidth, capacity,
                        cross_core_wait=0, same_core_wait=0):
    """执行确定性的单核评估并返回完整结果。

    带宽与容量由 CLI 从 config.txt 读出后传入，本函数不再提供配置默认值。
    """
    plan = build_singlecore_plan(graph_json)
    result = evaluate_scene_a(
        graph_json,
        plan,
        bandwidth=bandwidth,
        capacity=capacity,
        cross_core_wait=cross_core_wait,
        same_core_wait=same_core_wait,
    )
    result['execution_mode'] = 'singlecore'
    result['input_plan'] = None
    return result


@cli_errors
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='评估单核执行时间；无需多核调度结果文件')
    parser.add_argument('graph', help='原始计算图 JSON')
    parser.add_argument('--config', help='config.txt；默认使用计算图同目录配置')
    parser.add_argument('-o', '--output', help='评估结果 JSON 路径')
    parser.add_argument('--trace-output', help='Perfetto Trace JSON 路径')
    parser.add_argument('--log-output', help='精简结果日志路径')
    args = parser.parse_args(argv)
    if args.config and not os.path.isfile(args.config):
        raise FileNotFoundError(f'configuration file not found: {args.config}')

    case_path, _ = os.path.splitext(args.graph)
    config_path = args.config or os.path.join(
        os.path.dirname(args.graph), 'config.txt')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            'configuration file not found: {} (defaults to <graph dir>/config.txt; '
            'pass --config to choose another file)'.format(config_path))
    output_path = args.output or case_path + '_singlecore_res.json'
    trace_path = args.trace_output or case_path + '_singlecore_trace.json'
    log_path = args.log_output or case_path + '_singlecore_log.txt'

    graph = _read_json(args.graph)
    # 容量与带宽统一来自 config.txt，不再使用代码内置默认值。
    settings = read_evaluation_config(config_path)
    waits = read_scene_a_config(config_path)
    result = evaluate_singlecore(
        graph,
        bandwidth=settings['bandwidth'],
        capacity=settings['capacity'],
        cross_core_wait=waits['task_cross_core_wait_cycles'],
        same_core_wait=waits['task_same_core_wait_cycles'],
    )

    result['input_graph'] = os.path.basename(args.graph)
    write_text(output_path, json.dumps(
        result, ensure_ascii=False, indent=2) + '\n')
    write_text(trace_path, format_scene_a_trace_json(
        os.path.basename(args.graph), result) + '\n')
    write_text(log_path, format_multicore_result_log(
        os.path.basename(args.graph), result))
    emit(format_data_movement_log(result))
    emit('OK: singlecore makespan={:,} cycles; result={}; trace={}; log={}'.format(
        result['makespan'], output_path, trace_path, log_path))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
