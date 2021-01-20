import random
import networkx as nx
import itertools
import copy
import numpy as np

def cover_time(g, iter=1000):
    sum = 0
    for i in range(iter):
        t = 1
        s = random.choice(list(g.nodes))
        visited = [s]
        while len(visited) != len(g.nodes):
            s = random.choice(list(g.neighbors(s)))
            if s not in visited:
                visited.append(s)
            t += 1
        sum += t
    return sum / iter

def conductance(g):
    nodes = list(g.nodes)
    min = nx.conductance(g, [nodes[0]])
    for n in range(int(g.number_of_nodes() / 2)):
        cuts = itertools.combinations(nodes, n)
        for c in cuts:
            if c:
                cond = nx.conductance(g, c)
                min = cond if cond < min else min
    return min

def diameter(g):
    return nx.algorithms.distance_measures.diameter(g)

def connectivity(g):
    return nx.algebraic_connectivity(g)

def eigenvalue_n(g, n=0):
    L = nx.normalized_laplacian_matrix(g)
    e = np.linalg.eigvals(L.A)
    return e[n]

def max_eigenvalue(g, n=0):
    L = nx.normalized_laplacian_matrix(g)
    e = np.linalg.eigvals(L.A)
    return max(e)

def min_eigenvalue(g, n=0):
    L = nx.normalized_laplacian_matrix(g)
    e = np.linalg.eigvals(L.A)
    return max(e)
