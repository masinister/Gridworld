import networkx as nx
import copy
from random import sample

def fourrooms(w,h):
    g = nx.grid_graph(dim = (w,h))
    walls = [(i, int(h/2)) for i in range(w)] + [(int(w/2), j) for j in range(h)]
    doors = [(int(w/4), int(h/2)), (int(3 * w/4), int(h/2)), (int(w/2), int(h/4)), (int(w/2), int(3 * h/4))]
    for d in doors:
        walls.remove(d)
    edges_to_remove = g.edges(walls)
    g.remove_edges_from(copy.deepcopy(edges_to_remove))
    return g

def threerooms(w,h):
    g = nx.grid_graph(dim = (w,h))
    walls = [(i, int(h/3)) for i in range(w)] + [(i, int(2 * h/3)) for i in range(w)]
    doors = [(int(2 * w/3), int(h/3)), (int(w/3), int(2 * h/3))]
    for d in doors:
        walls.remove(d)
    edges_to_remove = g.edges(walls)
    g.remove_edges_from(copy.deepcopy(edges_to_remove))
    return g

def tworooms(w,h):
    g = nx.grid_graph(dim = (w,h))
    walls = [(i, int(h/2)) for i in range(w)]
    doors = [(int(w/2), int(h/2))]
    for d in doors:
        walls.remove(d)
    edges_to_remove = g.edges(walls)
    g.remove_edges_from(copy.deepcopy(edges_to_remove))
    return g

def oneroom(w,h):
    g = nx.grid_graph(dim = (w,h))
    return g

def randomwalls(w,h):
    g = nx.grid_graph(dim = (w,h))
    t = copy.deepcopy(g)
    random_nodes = sample([n for n in list(g.nodes()) if n not in [(0,0), (w-1,h-1)]], int(w * h / 3))
    for node in random_nodes:
        t.remove_node(node)
        if not nx.is_connected(t):
            break
        g.remove_edges_from(copy.deepcopy(g.edges(node)))
    return g
