import numpy as np
import pickle
import random

def neg_sample(all_ids, cnt, seed=0):
    random.seed(seed)
    with open("dag_dict.pkl", 'rb') as f:
        dag_dict = pickle.load(f)
    topo_arr = np.load('topological_arr.npy')
    
    samples = []
    available = {}
    for i in all_ids:
        idx, = np.where(topo_arr == i)[0]
        available[i] = list(set(topo_arr[idx:]) - dag_dict.get(i, set()))
    for i in range(cnt):
        x = random.choice(list(all_ids))
        samples.append([x, random.choice(available[x])])
    samples = np.array(samples, dtype=int)

    return samples 

def two_steps(D, all_ids, cnt=0, seed=0):
    random.seed(seed)
    D_two_step = {}
    total = []
    for k, v in D.items():
        if k not in all_ids: continue
        if k not in D_two_step.keys():
            D_two_step[k] = set()
        for i in v:
            D_two_step[k] |= D.get(i, set())
        D_two_step[k] -= D[k]
        total.append(list(map(list, zip([k for j in range(len(D_two_step[k]))], D_two_step[k]))))
    total = [a for b in total for a in b]
    if cnt == 0:
        cnt = len(total)
    samples = random.sample(total, cnt)

    return samples
