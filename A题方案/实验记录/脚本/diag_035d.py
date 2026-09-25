import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
out = os.path.join(os.environ['TEMP'], 'mm_extract', 'diag035')
main = json.load(open(os.path.join(out, 'case_035_d.json'), encoding='utf-8'))
core = main['per_core_timeline'][2]
want = set(int(x) for x in sys.argv[1:])
for o in sorted(core['ops'], key=lambda o: o['start']):
    if o['subgraph_id'] in want:
        print(o['subgraph_id'], o['op_id'], o['op'], o['pipe'], o['start'], o['end'])
# 367 在 MTE3 上的 COPY_OUT
mte3 = [o for o in core['ops'] if o['pipe'] == 'PIPE_MTE3']
mte3.sort(key=lambda o: o['start'])
print('核2 MTE3 在 24000-52000 的 COPY_OUT:', [(o['subgraph_id'], o['start'], o['end']) for o in mte3
                                             if 24000 < o['start'] < 52000][:20])
s3 = main['step3_by_core'][2] if isinstance(main['step3_by_core'], list) else main['step3_by_core'].get('2')
print(type(s3), list(s3)[:10] if isinstance(s3, dict) else str(s3)[:300])
