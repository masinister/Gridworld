import random
import networkx as nx
import itertools
import copy
import numpy as np

# TOO SLOW
def conductance(g, s=(0,0), t=(10,10), max_cut = 5):
    nodes = list(g.nodes)
    min = nx.conductance(g, [nodes[0]])
    for n in range(max_cut):
        cuts = itertools.combinations(nodes, n)
        for c in cuts:
            if c and ((s in c and t not in c) or (s not in c and t in c)):
                cond = nx.conductance(g, c)
                min = cond if cond < min else min
    return min

def cover_time(g, iter=1000):
    sum = 0
    for i in range(iter):
        t = 1
        s = (0,0)
        visited = [s]
        while len(visited) != len(g.nodes):
            s = random.choice(list(g.neighbors(s)))
            if s not in visited:
                visited.append(s)
            t += 1
        sum += t
    return sum / iter

def diameter(g):
    return nx.diameter(g)

def connectivity(g):
    return nx.algebraic_connectivity(g)

def eigenvalue_n(g, n=0):
    L = nx.normalized_laplacian_matrix(g)
    e = np.linalg.eigvals(L.A)
    return np.real(e[n])

def max_eigenvalue(g):
    L = nx.normalized_laplacian_matrix(g)
    e = np.linalg.eigvals(L.A)
    return np.real(max(e))

def min_eigenvalue(g):
    L = nx.normalized_laplacian_matrix(g)
    e = np.linalg.eigvals(L.A)
    return np.real(min(e))

def avg_eccentricity(g):
    return np.mean(list(nx.eccentricity(g).values()))

def efficiency(g):
    return nx.global_efficiency(g)

def closeness_vitality(g):
    v = list(nx.closeness_vitality(g).values())
    return np.mean(v[v != -np.inf])
