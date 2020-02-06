import numpy as np
import glob
import sys
import os

dataset = sys.argv[1]

res = []
for i in range(100):
    ofile = str(i) + '.txt'
    with open(os.path.join(dataset, ofile), 'r') as f:
        line = f.readlines()[-1]
        line = line.strip().split()[-1]
        res.append(float(line))

res = np.array(res)
print('mean:', np.mean(res))
print('max:', np.max(res))
print('min:', np.min(res))
print('std:', np.std(res))
