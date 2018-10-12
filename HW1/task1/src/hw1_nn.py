import numpy as np
import pickle
import argparse
import sys
import random
from scipy import spatial
import ipdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data

threshold = 0.1

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
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

D = {}

class Net(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.emb = nn.Linear(input_dim, embed_dim).cuda()
        self.dropout1 = nn.Dropout(0.5).cuda()
        self.lin = nn.Linear(2*embed_dim, 50).cuda()
        self.dropout2 = nn.Dropout(0.5).cuda()
        self.out = nn.Linear(50, 2).cuda()
        self.relu = nn.ReLU().cuda()

    def forward(self, x1, x2):
        x1 = self.emb(x1)
        x2 = self.emb(x2)
        x = torch.cat((x1, x2), 1)
        x = self.dropout1(x)
        x = self.lin(x)
        x = self.dropout2(x)
        x = self.out(x)
        x = self.relu(x)

        return x


def main():
    for i in range(len(train)):
        D[train[i][0]] = set()
    for i in range(len(train)):
        D[train[i][0]].add(train[i][1])

    for (a, b) in test_seen:
        if a not in D.keys():
            D[a] = set()
        else:
            D[a].add(b)

    if args.create_neg:
        neg_percentage = 0.5
        neg = np.zeros((int(len(train)*neg_percentage//(1-neg_percentage)),2))
        all_ids = set(np.arange(1, int(max(D.keys()))))
        for i in range(len(neg)):
            neg[i][0] = train[i][0]
            neg[i][1] = random.sample(all_ids - D[train[i][0]], 1)[0]
        print("negative samples:", len(neg))
        np.save("neg_sample.npy", neg)
    else:
        neg = np.load("neg_sample.npy")

    if args.train:
        all_train = np.vstack((train, neg))
        labels = np.zeros(len(all_train))
        labels[:len(train)] = 1
        print(train.shape, neg.shape, all_train.shape)
        order = np.arange(len(all_train))
        
        net = Net(int(max(D.keys())), 300)
        print(net)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
        loss_func = torch.nn.CrossEntropyLoss()


        epoch = 1
        batch_size = 300
        for e in range(epoch):
            random.shuffle(order)
            for b in range(len(all_train)//batch_size):
                loss = 0
                train_id_x1 = np.array(all_train[order][b*batch_size:(b+1)*batch_size][:,0], dtype=int)
                train_id_x2 = np.array(all_train[order][b*batch_size:(b+1)*batch_size][:,1], dtype=int)
                train_y = labels[order][b*batch_size:(b+1)*batch_size]
                train_x1 = np.zeros((batch_size, max(D.keys())))
                train_x2 = np.zeros((batch_size, max(D.keys())))
                for i in range(batch_size):
                    try:
                        train_x1[i][list(D[train_id_x1[i]])] = 1
                    except KeyError:
                        train_x1[i][train_id_x1[i]] = 1
                    try:
                        train_x2[i][list(D[train_id_x2[i]])] = 1
                    except KeyError:
                        train_x2[i][train_id_x2[i]] = 1
                    
                train_x1 = Variable(torch.from_numpy(train_x1)).float().cuda()
                train_x2 = Variable(torch.from_numpy(train_x2)).float().cuda()

                out = net(train_x1, train_x2)
                loss += loss_func(out, torch.LongTensor(train_y).cuda())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print("epoch: {} || batch: {} || loss: {}".format(e, b, loss))
        
        torch.save(net.state_dict(), 'model.mdl')
    
    if args.test:
        net = Net(int(max(D.keys())), 300)
        net.load_state_dict(torch.load('model.mdl'))
        f = open('nnpred.txt', 'w')
        for (a, b) in test:
            if (a not in D.keys()) or (b not in D.keys()):
                print('0', file=f)
            else:
                tmp_a = np.zeros(max(D.keys()))
                tmp_b = np.zeros(max(D.keys()))
                tmp_a[list(D[a])] = 1
                tmp_b[list(D[b])] = 1
                tmp_a = Variable(torch.from_numpy(tmp_a.reshape(1,-1))).float().cuda()
                tmp_b = Variable(torch.from_numpy(tmp_b.reshape(1,-1))).float().cuda()
                res = net(tmp_a, tmp_b)
                if res[0][1] > res[0][0]:
                    print('1', file=f)
                else:
                    print('0', file=f)


if __name__ == "__main__":
    main()
