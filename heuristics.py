import random
import networkx as nx
import itertools
import copy

def cover_time(g, iter=1000):
    g = copy.deepcopy(g)
    walls = [n for n in list(g.nodes) if g.degree(n)==0]
    g.remove_nodes_from(walls)
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
    g = copy.deepcopy(g)
    walls = [n for n in list(g.nodes) if g.degree(n)==0]
    g.remove_nodes_from(walls)
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
    g = copy.deepcopy(g)
    walls = [n for n in list(g.nodes) if g.degree(n)==0]
    g.remove_nodes_from(walls)
    return nx.algorithms.distance_measures.diameter(g)
