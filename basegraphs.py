import networkx as nx
import copy

def fourrooms(w,h):
    g = nx.grid_graph(dim = (w,h))
    walls = [(i, int(h/2)) for i in range(w)] + [(int(w/2), j) for j in range(h)]
    doors = [(int(w/4), int(h/2)), (int(3 * w/4), int(h/2)), (int(w/2), int(h/4)), (int(w/2), int(3 * h/4))]
    for d in doors:
        walls.remove(d)
    edges_to_remove = g.edges(walls)
    g.remove_edges_from(copy.deepcopy(edges_to_remove))
    return g
