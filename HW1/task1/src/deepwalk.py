import numpy as np
import pickle
import sys
from scipy import spatial
import ipdb
import random
from gensim.models import Word2Vec
import argparse
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from negative_sampling import neg_sample, two_steps
from scipy import spatial

threshold = 0.1
w2v_model_path = 'dim128-win5-sg1-hs1.w2v'
neg_sample_path = 'neg_sample_50.npy'

train_ids = set()

with open('data/t1-train.txt', 'r') as f:
    train = np.array(f.read().split('\n')[:-1])
    train = np.asarray(list(np.core.defchararray.chararray.split(train,' ')), dtype=int)

with open('data/t1-test-seen.txt', 'r') as f:
    test_seen = np.array(f.read().split('\n')[:-1])
    test_seen = np.asarray(list(np.core.defchararray.chararray.split(test_seen,' ')), dtype=int)

with open('data/t1-test.txt', 'r') as f:
    test = np.array(f.read().split('\n')[:-1])
    test = np.asarray(list(np.core.defchararray.chararray.split(test,' ')), dtype=int)

parser = argparse.ArgumentParser()
parser.add_argument('--create_neg', action='store_true')
parser.add_argument('--train_w2v', action='store_true')
parser.add_argument('--train_svm', action='store_true')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

D = {}

def random_walk(node, max_steps):
    path = [node]
    while len(path) < max_steps:
        cur = path[-1]
        if len(D.get(cur, set())) > 0:
            path.append(random.choice(list(D[cur])))
        else:   break
    return [str(i) for i in path]

def main():
    for i in range(len(train)):
        D[train[i][0]] = set()
    for i in range(len(train)):
        D[train[i][0]].add(train[i][1])
        train_ids.add(train[i][0])
        train_ids.add(train[i][1])

    for (a, b) in test_seen:
        if a not in D.keys():
            D[a] = set()
        else:
            D[a].add(b)

    if args.train_w2v:
        print("Training Word2Vec")
        num_paths = 80
        max_steps = 40
        node_order = list(D.keys())
        walks = []
        random.seed(0)
        for i in range(num_paths):
            print("Generating Walks...{}".format(i))
            random.shuffle(node_order)
            for node in node_order:
                walks.append(random_walk(node, max_steps))
        print("Training")
        embedding_dim = 128
        window_size = 10
        w2v = Word2Vec(walks, size=embedding_dim, window=window_size, min_count=0, sg=1, hs=1, workers=20)
        w2v.save(w2v_model_path)
    else:
        print("Loading Word2Vec")
        w2v = Word2Vec.load(w2v_model_path)


    if args.train_svm:
        if args.create_neg:
            print("Creating Negative Samples")
            random.seed(0)
            neg_percentage = 0.6
            neg_cnt = int(len(train)*neg_percentage // (1-neg_percentage))
            all_ids = train_ids 
            #neg = neg_sample(all_ids, neg_cnt)
            neg = two_steps(D, all_ids, neg_cnt)
            print("negative samples:", len(neg))
            #np.save(neg_sample_path, neg)
        else:
            print("Loading Negative Samples")
            neg = np.load(neg_sample_path)
        all_data = np.vstack((train, test_seen, neg))
        Y = np.zeros(len(all_data))
        Y[:-len(neg)] = 1
        X = []
        tmp = 0
        for a, b in all_data:
            if str(int(a)) in w2v.wv and str(int(b)) in w2v.wv:
                X.append(np.hstack((w2v.wv[str(int(a))], w2v.wv[str(int(b))])))
                tmp += 1
            else:
                Y = np.delete(Y, tmp)

        print("Training Linear SVC")
        #clf = LinearSVC(random_state=0)
        clf = RandomForestClassifier(n_jobs=20)
        clf.fit(X, Y)
        #joblib.dump(clf, 'RF.joblib')
    else:
        print("Loading Linear SVC")
        clf = joblib.load('RF.joblib')

    predict(w2v, clf)

def predict(w2v, clf):
    abc = 0
    print("Predicting")
    vecs = []
    res2 = []
    with open("mypred.txt", "w") as f:
        cnt = 0
        for i, (a, b) in enumerate(test):
            print("\r{}".format(i), end='')
            res = 1
            if (str(int(a)) not in w2v.wv) or (str(int(b)) not in w2v.wv):
                print("1", file=f)
                cnt += 1
                continue

            #vec = np.hstack((w2v.wv[str(int(a))], w2v.wv[str(int(b))]))
            #vecs.append(vec)
            if 1 - spatial.distance.cosine(w2v.wv[str(int(a))], w2v.wv[str(int(b))]) < 0.6:
                res = 0
                cnt += 1
            print(int(res), file=f)
                #res2.append(1)
                #abc += 1
            #else:
            #    res2.append(0)
            '''
            if i % 1000 == 0 or i == len(test) - 1:
                res = clf.predict(vecs)
                vecs = []
                cnt += sum(res)
                for a, j in enumerate(res):
                    #if res2[a] == 1 and j == 0:
                    #    j = 1
                    #    cnt += 1
                    print(int(j), file=f)
                res2 = []
            '''
        print("")    
        print(cnt, '/', len(test))
        print(abc)

if __name__ == "__main__":
    main()
