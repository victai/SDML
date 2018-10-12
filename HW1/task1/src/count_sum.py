import numpy as np
import sys

with open(sys.argv[1], 'r') as f:
    a = np.array(f.read().split('\n')[:-1], dtype=int)
    print(sum(a), "/", len(a))
