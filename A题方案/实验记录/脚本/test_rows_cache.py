"""复审问题3 的复现：同键 rows 但没有方案与评估文件时，不应复用；键不同也不应复用。"""
import json
import os
import sys
import tempfile

ROOT = r'C:\Users\richd\Desktop\数学建模\A题(1)\A题方案'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from solver.run_batch import cached_rows, code_fingerprint

code = code_fingerprint()
key = dict(code, cores=[1], problems=[1], verify=0, small=[0, 0], case='x')
row = {'case': 'case_test', 'problem': 1, 'cores': 1, 'eval_file': 'eval/case_test_p1_n1.json'}
with tempfile.TemporaryDirectory() as out:
    os.makedirs(os.path.join(out, 'rows'))
    done = os.path.join(out, 'rows', 'case_test.json')
    json.dump({'key': key, 'rows': [row]}, open(done, 'w'))
    print('文件缺失时复用:', cached_rows(done, key, out) is not None)
    os.makedirs(os.path.join(out, 'plans'))
    os.makedirs(os.path.join(out, 'eval'))
    open(os.path.join(out, 'plans', 'case_test_p1_n1.json'), 'w').write('{}')
    ev = os.path.join(out, 'eval', 'case_test_p1_n1.json')
    open(ev, 'w').write('{}')
    row['eval_file'] = ev
    json.dump({'key': key, 'rows': [row]}, open(done, 'w'))
    print('文件齐全、键一致时复用:', cached_rows(done, key, out) is not None)
    print('代码指纹变化时复用:', cached_rows(done, dict(key, solver='other'), out) is not None)
    json.dump({'key': {'cores': [1], 'problems': [1], 'verify': 0}, 'rows': [row]}, open(done, 'w'))
    print('旧格式键时复用:', cached_rows(done, key, out) is not None)
