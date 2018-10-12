import numpy as np
import pickle
import sys
from scipy import spatial
import ipdb

threshold = 0.1
confidence_threshold = 0.05

with open('data/t1-train.txt', 'r') as f:
    train = np.array(f.read().split('\n')[:-1])
    train = np.asarray(list(np.core.defchararray.chararray.split(train,' ')), dtype=int)

with open('data/t1-test-seen.txt', 'r') as f:
    test_seen = np.array(f.read().split('\n')[:-1])
    test_seen = np.asarray(list(np.core.defchararray.chararray.split(test_seen,' ')), dtype=int)

with open('data/t1-test.txt', 'r') as f:
    test = np.array(f.read().split('\n')[:-1])
    test = np.asarray(list(np.core.defchararray.chararray.split(test,' ')), dtype=int)

D = {}

def main():
    for i in range(len(train)):
        D[train[i][0]] = set()
        D[train[i][1]] = set()
    for i in range(len(train)):
        D[train[i][0]].add(train[i][1])
        D[train[i][1]].add(train[i][0])

    for (a, b) in test_seen:
        if a not in D.keys():
            D[a] = set()
        else:
            D[a].add(b)
        if b not in D.keys():
            D[b] = set()
        else:
            D[b].add(a)

    predict()
    
def predict():
    with open("mypred.txt", "w") as f:
        for (a, b) in test:
            if (a not in D.keys()) or (b not in D.keys()):
                continue
            else:
                if jaccard(D[a], D[b]) > confidence_threshold:
                    D[a].add(b)
                    D[b].add(a)
        cnt = 0
        for (a, b) in test:
            if (a not in D.keys()) or (b not in D.keys()):
                f.write("1\n")
                cnt += 1
            else:
                if jaccard(D[a], D[b]) > 0.05:
                    f.write("1\n")
                    cnt += 1
                else:
                    f.write("0\n")

        '''
        for (a, b) in test:
            if (a not in D.keys()) or (b not in D.keys()):
                f.write("1\n")
            else:
                L2_a = set()
                L2_b = set()
                for i in D[a]:
                    try:
                        L2_a |= D[i]
                    except KeyError:
                        pass
                for i in D[b]:
                    try:
                        L2_b |= D[i]
                    except KeyError:
                        pass
                #print("L1_a, L1_b: ", jaccard(D[a],D[b]))
                #print("L2_a, L2_b: ", jaccard(L2_a,L2_b))
                score1 = jaccard(D[a], D[b])
                score = score1 + \
                        jaccard(D[a], L2_b) * 0.5 + \
                        jaccard(L2_a, D[b]) * 0.5 + \
                        jaccard(L2_a, L2_b) * 0.25
                #score = cosine_similarity(D[a], D[b]) + \
                #        cosine_similarity(D[a], L2_b) * 0.5 + \
                #        cosine_similarity(L2_a, D[b]) * 0.5 + \
                #        cosine_similarity(L2_a, L2_b) * 0.25
                #print(score)
                if score1 > 0.025 or score > 0.1 or common_neighbor(D[a], D[b]) >= 2:
                    f.write("1\n")
                    cnt += 1
                else:
                    f.write("0\n")
        '''
        print(cnt, '/', len(test))

def cosine_similarity(a, b):
    tmp_a = np.zeros(max(D.keys()))
    tmp_b = np.zeros(max(D.keys()))
    tmp_a[list(a)] = 1
    tmp_b[list(b)] = 1
    return (1 - spatial.distance.cosine(tmp_a, tmp_b))

def jaccard(a, b):
    try:
        return (len(a & b) / len(a | b))
    except ZeroDivisionError:
        return 0

def common_neighbor(a, b):
    return len(a & b)

if __name__ == "__main__":
    main()
