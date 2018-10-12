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
import xgboost
from xgboost import XGBClassifier
from negative_sampling import neg_sample, two_steps
from scipy import spatial

threshold = 0.1
model_path = "PRUNE/graph.embeddings"

train_ids = set()
all_ids = set()

with open('data/t1-train.txt', 'r') as f:
    train = np.array(f.read().split('\n')[:-1])
    train = np.asarray(list(np.core.defchararray.chararray.split(train,' ')), dtype=int)

with open('data/t1-test-seen.txt', 'r') as f:
    test_seen = np.array(f.read().split('\n')[:-1])
    test_seen = np.asarray(list(np.core.defchararray.chararray.split(test_seen,' ')), dtype=int)

with open('data/t1-test.txt', 'r') as f:
    test = np.array(f.read().split('\n')[:-1])
    test = np.asarray(list(np.core.defchararray.chararray.split(test,' ')), dtype=int)
with open('degree.txt', 'r') as f:
    degree = np.array(f.read().split('\n')[:-1], dtype=int)

parser = argparse.ArgumentParser()
parser.add_argument('--create_neg', action='store_true')
parser.add_argument('--train_w2v', action='store_true')
parser.add_argument('--train_svm', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--validate', action='store_true')
args = parser.parse_args()

D = {}
D_test = {}

def main():
    for i in range(len(train)):
        D[train[i][0]] = set()
    for i in range(len(train)):
        D[train[i][0]].add(train[i][1])
        train_ids.add(train[i][0])
        train_ids.add(train[i][1])
        all_ids.add(train[i][0])
    for i in range(len(test)):
        D_test[test[i][0]] = set()
    for i in range(len(test)):
        D_test[test[i][0]].add(test[i][1])
        all_ids.add(test[i][0])

    for (a, b) in test_seen:
        all_ids.add(a)
        train_ids.add(a)
        train_ids.add(b)
        if a not in D.keys():
            D[a] = set()
        else:
            D[a].add(b)

    print("Loading Embedding")
    emb = np.genfromtxt(model_path, delimiter=',')

    if args.train_svm:
        print("Creating Negative Samples")
        random.seed(0)
        neg_percentage = 0.5
        neg_cnt = int(len(train)*neg_percentage // (1-neg_percentage))
        #all_ids = train_ids 
        #neg = neg_sample(all_ids, neg_cnt)
        neg = two_steps(D, D_test, all_ids, neg_cnt)
        print("len(train) =", len(train))
        print("negative samples:", len(neg))
        #np.save(neg_sample_path, neg)

        #all_data = np.vstack((train, test_seen, neg))
        a = 20000
        b = 5000
        c = a+b
        random.shuffle(train)
        random.shuffle(test_seen)
        random.shuffle(neg)
        if args.validate:
            all_train = np.vstack((train[:a], test_seen[:-b], neg[:-c]))
        else:
            all_train = np.vstack((train, test_seen, neg))
        all_val = np.vstack((train[-a:], test_seen[-b:], neg[-c:]))
        Y_train = np.zeros(len(all_train))
        Y_val = np.zeros(len(all_val))
        if args.validate:
            Y_train[:-(len(neg))] = 1
        else:
            Y_train[:-(len(neg)) - c] = 1
        Y_val[:-c] = 1
        X_train = []
        X_val = []
        for a, b in all_train:
            jaccard_coeff = 1
            if a in D.keys() and b in D.keys():
                jaccard_coeff = jaccard(D[a], D[b])
            X_train.append(np.hstack((emb[a], emb[b], jaccard_coeff, degree[a], degree[b])))
        for a, b in all_val:
            jaccard_coeff = 1
            if a in D.keys() and b in D.keys():
                jaccard_coeff = jaccard(D[a], D[b])
            X_val.append(np.hstack((emb[a], emb[b], jaccard_coeff, degree[a], degree[b])))

        '''
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
        '''
        print("Training Linear SVC")
        #clf = LinearSVC(random_state=0)
        #clf = RandomForestClassifier(n_jobs=20)
        
        
        params = {}
        params['objective'] = 'binary:hinge'
        params['eval_metric'] = 'logloss'
        params['eta'] = 0.04
        params['max_depth'] = 3
        params['learning_rate'] = 0.01

        d_train = xgboost.DMatrix(X_train, label=Y_train)
        d_valid = xgboost.DMatrix(X_val, label=Y_val)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        clf = xgboost.train(params, d_train, 100, watchlist, early_stopping_rounds=100, verbose_eval=10)
        y_pred = clf.predict(xgboost.DMatrix(X_val))
        print(y_pred)
        y_pred[y_pred>0.5] = 1
        y_pred[y_pred<=0.5] = 0
        print("positive correctness:", sum(y_pred[:c] == Y_val[:c]), '/', c)
        print("negative correctness:", sum(y_pred[c:] == Y_val[c:]), '/', c)
        #exit()
        
        
        #clf = XGBClassifier(n_jobs=1000, objective="binary:logistic")
        #clf.fit(np.array(X_train), np.array(Y_train))
        '''
        print("Validating")
        res = clf.predict(np.array(X_val))
        print(res)
        print("positive correctness:", sum(Y_val[:c] == res[:c]), '/', c)
        print("negative correctness:", sum(Y_val[c:] == res[c:]), '/', c)
        '''
        #exit()
        #joblib.dump(clf, 'RF.joblib')
    else:
        print("Loading Linear SVC")
        clf = joblib.load('RF.joblib')

    predict(emb, clf)

def predict(emb, clf):
    abc = 0
    print("Predicting")
    vecs = []
    res2 = []
    X_test = []
    with open("mypred.txt", "w") as f:
        cnt = 0
        for i, (a, b) in enumerate(test):
            res = 1
            
            jaccard_coeff = 1
            if a in D.keys() and b in D.keys():
                jaccard_coeff = jaccard(D[a], D[b])
            X_test.append(np.hstack((emb[a], emb[b], jaccard_coeff, degree[a], degree[b])))

        res = clf.predict(xgboost.DMatrix(X_test))
        #res = clf.predict(X_test)
        cnt = 0
        print(sum(res))
        missing = 0
        for i, (a, b) in enumerate(test):
            if a not in train_ids or b not in train_ids:
                print("1", file=f)
                cnt += 1
                missing += 1
            else:
                if res[i] > 0.5:
                    print("1", file=f)
                    cnt += 1
                else:
                    print("0", file=f)
        print(" ")    
        print('missing', missing)
        print(cnt, "/", len(res))
        print(cnt)


        #vec = np.hstack((w2v.wv[str(int(a))], w2v.wv[str(int(b))]))
        #vecs.append(vec)
        #if 1 - spatial.distance.cosine(w2v.wv[str(int(a))], w2v.wv[str(int(b))]) < 0.6:
        #    res = 0
        #    cnt += 1
        #print(int(res), file=f)
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
        print("")    
        print(cnt, '/', len(test))
        print(abc)
        '''


def jaccard(a, b):
    try:
        return (len(a & b) / len(a | b))
    except ZeroDivisionError:
        return 0

if __name__ == "__main__":
    main()
