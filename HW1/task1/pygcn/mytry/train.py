from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from utils import load_data, accuracy
from models import GCN
from layers import GraphConvolution

from scipy import spatial
from negative_sampling import two_steps
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--train_svm', action='store_true', default=False)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
D, adj, features, labels, idx_train, idx_val, test_data, train_data, train_ids = load_data()

class EMB(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(EMB, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        #self.gc2 = GraphConvolution(nhid, nclass)
        #self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = self.gc2(x, adj)
        #return F.log_softmax(x, dim=1)
        return x


# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=1,               ##changed
            dropout=args.dropout)
emb = EMB(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=1,               ##changed
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    #idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.mse_loss(output[idx_train].flatten(), labels[idx_train].float())
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    #print("output", output)
    #print("labels", labels)

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.mse_loss(output[idx_val].flatten(), labels[idx_val].float())
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    #emb = torch.nn.Sequential(*list(model.children()))
    pretrained_dict = model.state_dict()
    emb_dict = emb.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in emb_dict}
    emb_dict.update(pretrained_dict)
    emb.load_state_dict(emb_dict)

    emb.eval()
    emb.cuda()
    f = open('mypred.txt', 'w')
    threshold = 0.9
    cnt = 0
    out = emb(features, adj)
    tt = model(features, adj)
    out = out.detach().cpu().numpy()
    print(out.shape)
    #for a, b in train_data:
    #    sim = 1 - spatial.distance.cosine(out[a].detach().cpu().numpy(), out[b].detach().cpu().numpy())
    #    print(sim)

    w2v = Word2Vec.load("../../dim128-win5-sg1-hs1.w2v")
    if args.train_svm:
        print("Creating Negative Samples")
        random.seed(0)
        neg_percentage = 0.5
        neg_cnt = int(len(train_data)*neg_percentage // (1-neg_percentage))
        all_ids = train_ids 
        #neg = neg_sample(all_ids, neg_cnt)
        neg = two_steps(D, all_ids)
        print("negative samples:", len(neg))

        all_data = np.vstack((train_data, neg))
        Y = np.zeros(len(all_data))
        Y[:-len(neg)] = 1
        X = []
        tmp = 0
        for a, b in all_data:
            X.append(np.hstack((out[a], out[b])))

        print(len(X))
        print(X[0].shape)
        print("Training Linear SVC")
        #clf = LinearSVC(random_state=0)
        clf = RandomForestClassifier(n_jobs=8)
        clf.fit(X, Y)
        joblib.dump(clf, 'RF.joblib')
    else:
        print("Loading Linear SVC")
        clf = joblib.load('RF.joblib')
    
    vecs = []
    for i, (a, b) in enumerate(test_data):
        print("\r{}".format(i), end='')
       
        #if sum(adj.cpu().to_dense()[a]) == 0 or sum(adj.cpu().to_dense()[b]) == 0:
        if (a not in D.keys()) or (b not in D.keys()):
            print("1", file=f)
            cnt += 1
            continue
        #tmp = adj.cpu().to_dense()[a]
        #idx = np.nonzero(tmp.numpy())[0]
        #idx_x = [a for i in range(len(idx))]
        #tmp_a = torch.sparse.LongTensor((torch.LongTensor([idx_x, idx])), torch.ones(len(idx)), torch.Size([37501,37501]))
        #out_a = emb(features, tmp_a.cuda())
        #tmp = adj.cpu().to_dense()[b]
        #idx = np.nonzero(tmp.numpy())[0]
        #idx_x = [b for i in range(len(idx))]
        #tmp_b = torch.sparse.LongTensor((torch.LongTensor([idx_x, idx])), torch.ones(len(idx)), torch.Size([37501,37501]))
        #out_b = emb(features, tmp_b.cuda())
        '''
        sim = 1 - spatial.distance.cosine(out[a].detach().cpu().numpy(), out[b].detach().cpu().numpy())
        if sim > threshold:
            print("1", file=f)
            cnt += 1
        else:
            print("0", file=f)
        '''
        vecs.append(np.hstack((out[a], out[b])))    

        if i % 1000 == 0 or i == len(test_data) - 1:
            res = clf.predict(vecs)
            vecs = []
            cnt += sum(res)
            for j in res:
                print(int(j), file=f)
    print("")
    print(cnt, "/", len(test_data))

    #loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    #acc_test = accuracy(output[idx_test], labels[idx_test])
    #print("Test set results:",
    #      "loss= {:.4f}".format(loss_test.item()),
    #      "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
if args.train:
    for epoch in range(args.epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    torch.save(model.state_dict(), 'model.mdl')
else:
    model.load_state_dict(torch.load('model.mdl'))

# Extract Embedding
#emb = torch.nn.Sequential(*list(model.children())[:-1])

# Testing
test()
