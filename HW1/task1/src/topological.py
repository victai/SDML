import numpy as np
import ipdb

def cyclic(graph):
    '''
    visited = set()
    path = [object()]
    path_set = set(path)
    stack = [iter(graph)]
    while stack:
        for v in stack[-1]:
            print(v)
            if v in path_set:
                return True
            elif v not in visited:
                visited.add(v)
                path.append(v)
                path_set.add(v)
                stack.append(iter(graph.get(v, ())))
                break
        else:
            path_set.remove(path.pop())
            stack.pop()
    return False
    '''
    path = set()
    visited = set()

    def visit(vertex):
        if vertex in visited:
            return False
        visited.add(vertex)
        path.add(vertex)
        for neighbour in graph.get(vertex, ()):
            if neighbour in path:
                graph[vertex].remove(neighbour)
                #print(vertex, neighbour)
                return True
            if visit(neighbour):
                return True
        path.remove(vertex)
        return False
    #for v in graph:
    #    print(v, visit(v))
    return any(visit(v) for v in graph)


with open('data/t1-train.txt', 'r') as f:
    train = np.array(f.read().split('\n')[:-1])
    train = np.asarray(list(np.core.defchararray.chararray.split(train,' ')), dtype=int)

with open('data/t1-test-seen.txt', 'r') as f:
    test_seen = np.array(f.read().split('\n')[:-1])
    test_seen = np.asarray(list(np.core.defchararray.chararray.split(test_seen,' ')), dtype=int)
D = {}

for i in range(len(train)):
    D[train[i][0]] = set()
for i in range(len(train)):
    D[train[i][0]].add(train[i][1])

for (a, b) in test_seen:
    if a not in D.keys():
        D[a] = set()
    else:
        D[a].add(b)

cnt = 0
while cyclic(D):
    cnt += 1

from collections import deque

GRAY, BLACK = 0, 1

def topological(graph):
    order, enter, state = deque(), set(graph), {}

    def dfs(node):
        state[node] = GRAY
        for k in graph.get(node, ()):
            sk = state.get(k, None)
            if sk == GRAY: raise ValueError("cycle")
            if sk == BLACK: continue
            enter.discard(k)
            dfs(k)
        order.appendleft(node)
        state[node] = BLACK

    while enter: dfs(enter.pop())
    return order

graph2 = {
    "a": ["b", "c", "d"],
    "b": [],
    "c": ["d"],
    "d": [],
    "e": ["g", "f", "q"],
    "g": [],
    "f": [],
    "q": []
}

try:    T = np.array(topological(D))
except ValueError: print ("Cycle!")

import pickle
with open("dag_dict.pkl", 'wb') as f:
    pickle.dump(D, f)
np.save("topological_arr.npy", T)
