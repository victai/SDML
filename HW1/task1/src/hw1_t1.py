import numpy as np
import pickle
import sys
from scipy import spatial

sys.setrecursionlimit(1000000)
ID_MAX = 30001
simi_thresh = 0.95
traced = np.zeros(ID_MAX, dtype=int)

with open('t1-train.txt', 'r') as f:
    train = np.array(f.read().split('\n')[:-1])
train = np.asarray(list(np.core.defchararray.chararray.split(train,' ')), dtype=int)

with open('t1-test-seen.txt', 'r') as f:
    test_seen = np.array(f.read().split('\n')[:-1])
test_seen = np.asarray(list(np.core.defchararray.chararray.split(test_seen,' ')), dtype=int)

with open('t1-test.txt', 'r') as f:
    test = np.array(f.read().split('\n')[:-1])
test = np.asarray(list(np.core.defchararray.chararray.split(test,' ')), dtype=int)

def main():
    global traced
    
    train_cite = np.zeros((ID_MAX,ID_MAX), dtype=np.float16)
    for (a, b) in train:
        train_cite[a][b] = 1
    for i in range(2):
        if i in train[:,0]:
            print(i)
            reach(train_cite, i, 0)
            traced[i] = 1
    with open("train_cite_depth.pkl", "wb") as f:
        pickle.dump(train_cite, f)
    
    predict()
    
def predict():
    with open("train_cite.pkl", "rb") as f:
        cite = pickle.load(f)

    for (a, b) in test_seen:
        cite[a][b] = 1
        cite[a] = np.logical_or(cite[a], cite[b])

    with open("mypred.txt", "w") as f:
        for (a, b) in test:
            if (1 - spatial.distance.cosine(cite[a], cite[b])) > simi_thresh:
                f.write("1\n")
            else:
                f.write("0\n")

def reach(train_cite, node, depth):
    if depth > 3: return
    for i in range(ID_MAX):
        if train_cite[node][i] == 1:
            print(node, i, depth)
            print(train_cite[1210][1640])
            if (node == 691):
                train_cite[1210][1640] = 1
            #ipdb.set_trace()
            if traced[i] == 0 and train_cite[node][i] == 1:
                reach(train_cite, i, depth+1)
            #train_cite[node] = np.logical_or(train_cite[node], train_cite[i])
            mask1 = np.array(train_cite[i],dtype=bool)
            mask2 = np.logical_or(np.array(train_cite[i],dtype=bool), np.logical_not(train_cite[node]))
            train_cite[node][mask1] = np.minimum(train_cite[node], train_cite[i]+1)[mask1]
            train_cite[node][mask2] = (train_cite[i]+1)[mask2]


if __name__ == "__main__":
    main()
