import numpy as np
import scipy.sparse as sp
import torch
from gensim.models import Word2Vec

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    with open('../../data/t1-train.txt', 'r') as f:
        train_data = np.array(f.read().split('\n')[:-1])
        train_data = np.asarray(list(np.core.defchararray.chararray.split(train_data,' ')), dtype=int)

    with open('../../data/t1-test-seen.txt', 'r') as f:
        test_seen_data = np.array(f.read().split('\n')[:-1])
        test_seen_data = np.asarray(list(np.core.defchararray.chararray.split(test_seen_data,' ')), dtype=int)

    with open('../../data/t1-test.txt', 'r') as f:
        test_data = np.array(f.read().split('\n')[:-1])
        test_data = np.asarray(list(np.core.defchararray.chararray.split(test_data,' ')), dtype=int)

    D = {}
    train_ids = set()
    for i in range(len(train_data)):
        D[train_data[i][0]] = set()
    for i in range(len(train_data)):
        D[train_data[i][0]].add(train_data[i][1])
        train_ids.add(train_data[i][0])
        train_ids.add(train_data[i][1])

    for (a, b) in test_seen_data:
        if a not in D.keys():
            D[a] = set()
        else:
            D[a].add(b)
    max_id = int(max(D.keys()))
    
    train_data = np.vstack((train_data, test_seen_data))

    adj = sp.coo_matrix((np.ones(len(train_data)), (train_data[:,0], train_data[:,1])),
                            shape=(max_id+1, max_id+1))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    with open("degree.txt", 'r') as f:
        labels = torch.from_numpy(np.array(f.read().split('\n')[:-1], dtype=float))
    #features = sp.coo_matrix((np.ones(max_id+1), (np.arange(max_id+1), np.arange(max_id+1))),
    #                            shape=(max_id+1, max_id+1))
    #features = sparse_mx_to_torch_sparse_tensor(features)

    w2v = Word2Vec.load("../../dim128-win5-sg1-hs1.w2v")
    features = np.zeros((max_id+1, 128))
    for i in range(1, max_id+1):
        if str(i) in w2v.wv:
            features[i] = w2v.wv[str(i)]
    features = torch.FloatTensor(features)
    
    idx_train = torch.LongTensor(range(int(max_id * 0.9)))
    idx_val = torch.LongTensor(range(int(max_id * 0.9), max_id))

    return D, adj, features, labels, idx_train, idx_val, test_data, train_data, train_ids
    
    """Load citation network dataset (cora only for now)"""
    '''
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    return adj, features, labels, idx_train, idx_val, idx_test
    '''

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
