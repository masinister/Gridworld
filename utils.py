import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

def n_regular_bipartite(size = 20, n = 4):
    G = nx.Graph()
    G.add_nodes_from(list(range(size)), bipartite = 0)
    G.add_nodes_from(list(range(size + 1, size + size)), bipartite = 1)
    P = np.transpose([list(np.random.permutation(size)) for _ in range(n)])
    while np.any([len(set(p)) != len(p) for p in P]):
        P = np.transpose([list(np.random.permutation(size)) for _ in range(n)])
    for i in range(size):
        G.add_edges_from([(size + i, p) for p in P[i]])
    return G

def add_random_edges(G, n):
    non_edges = list(nx.non_edges(G))
    prob = [1 if nx.has_path(G, e[0], e[1]) else 0 for e in non_edges]
    prob /= np.sum(prob)
    edge_indices = np.random.choice(len(non_edges), n, p = prob)
    for i in edge_indices:
        G.add_edge(non_edges[i][0],non_edges[i][1])

def add_kleinberg_edges(G, n, r):
    non_edges = list(nx.non_edges(G))
    prob = [nx.shortest_path_length(G, e[0], e[1])**(-r) if nx.has_path(G, e[0], e[1]) else 0 for e in non_edges]
    prob /= np.sum(prob)
    edge_indices = np.random.choice(len(non_edges), n, p = prob)
    for i in edge_indices:
        G.add_edge(non_edges[i][0],non_edges[i][1])
