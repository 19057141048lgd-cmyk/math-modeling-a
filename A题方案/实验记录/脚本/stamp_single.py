import glob
import os
import sys

ROOT = r'C:\Users\richd\Desktop\数学建模\A题(1)\A题方案'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from solver.config import case_path
from solver.run_batch import RESULTS, code_fingerprint, digest

code = code_fingerprint()
n = 0
for res in glob.glob(os.path.join(RESULTS, 'single', '*_single.json')):
    case = os.path.basename(res)[:-len('_single.json')]
    with open(res[:-5] + '.key', 'w', encoding='utf-8') as f:
        f.write('%s-%s' % (code['official'], digest([case_path(case)])))
    n += 1
print('写入', n, '个 key，official =', code['official'])
