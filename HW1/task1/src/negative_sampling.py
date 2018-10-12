import numpy as np
import pickle
import random
import ipdb

def neg_sample(all_ids, cnt, seed=0):
    random.seed(seed)
    with open("dag_dict.pkl", 'rb') as f:
        dag_dict = pickle.load(f)
    topo_arr = np.load('topological_arr.npy')
    
    samples = []
    available = {}
    for i in all_ids:
        try:
            idx, = np.where(topo_arr == i)[0]
        except ValueError:
            continue
        available[i] = list(set(topo_arr[idx:]) - set([i]) - dag_dict.get(i, set()))
    steps = 5
    for i in range(cnt):
        x = random.choice(list(all_ids))
        #samples.append([x, random.choice(available[x][:steps])])
        for j in range(steps):
            if j >= len(available.get(x, set())):  break
            samples.append([x, available[x][j]])
    print(len(samples))
    return samples 

def two_steps(D, D_test, all_ids, cnt=0, seed=0):
    random.seed(seed)
    if cnt == 0:
        cnt = len(total)
    D_two_step = {}
    D_three_step = {}
    D_four_step = {}
    two_total = []
    three_total = []
    four_total = []
    for k, v in D.items():
        if k not in all_ids: continue
        if k not in D_two_step.keys():
            D_two_step[k] = set()
        for i in v:
            D_two_step[k] |= D.get(i, set())
        D_two_step[k] -= D[k]
        D_two_step[k] -= D_test.get(k, set())
        two_total.append(list(map(list, zip([k for j in range(len(D_two_step[k]))], D_two_step[k]))))
    two_total = [a for b in two_total for a in b]
    print(len(two_total))
    print("2 done")
    for k, v in D_two_step.items():
        if k not in all_ids: continue
        if k not in D_three_step.keys():
            D_three_step[k] = set()
        for i in v:
            D_three_step[k] |= D.get(i, set())
        D_three_step[k] -= D[k]
        D_three_step[k] -= D_two_step[k]
        D_three_step[k] -= D_test.get(k, set())
        three_total.append(list(map(list, zip([k for j in range(len(D_three_step[k]))], D_three_step[k]))))
    three_total = [a for b in three_total for a in b]
    print(len(three_total))
    print("3 done")

    for k, v in D_three_step.items():
        if k not in all_ids: continue
        if k not in D_four_step.keys():
            D_four_step[k] = set()
        for i in v:
            D_four_step[k] |= D.get(i, set())
        D_four_step[k] -= D[k]
        D_four_step[k] -= D_two_step[k]
        D_four_step[k] -= D_three_step[k]
        D_four_step[k] -= D_test.get(k, set())
        four_total.append(list(map(list, zip([k for j in range(len(D_four_step[k]))], D_four_step[k]))))
    four_total = [a for b in four_total for a in b]
    print(len(four_total))
    print("4 done")
    
    two_cnt = int(0.4*cnt)
    three_cnt = int(0.4*cnt)
    four_cnt = int(0.2*cnt)
    total = []
    total += random.sample(two_total, two_cnt)
    total += random.sample(three_total, three_cnt)
    total += random.sample(four_total, four_cnt)
    #return total
    '''
    total.append(list(map(list, zip([k for j in range(len(D_two_step[k]))], D_two_step[k]))))
    total = [a for b in total for a in b]
    #samples = random.sample(total, int(cnt))
    samples = neg_sample(all_ids, cnt)
    '''
    
    print("start")
    samples2 = neg_sample(all_ids, int(cnt*0.2))
    print("end")
    total += random.sample(samples2, int(cnt*0.2))
    total = np.unique(total, axis=0).tolist()
    
    return total
