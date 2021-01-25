import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from heuristics import cover_time, diameter, connectivity, eigenvalue_n, max_eigenvalue, avg_eccentricity

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

def params(g):
    g = copy.deepcopy(g)
    walls = [n for n in list(g.nodes) if g.degree(n)==0]
    g.remove_nodes_from(walls)
    return diameter(g), avg_eccentricity(g), connectivity(g), max_eigenvalue(g)
