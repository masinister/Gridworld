import gym
import gym_graphworld
import networkx as nx
import copy
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from mc import epsilon_greedy, q_learning, q_learning_mm, sarsa, sarsa_mm
from utils import add_kleinberg_edges, add_random_edges

d = (11,11)
basegraph = nx.grid_graph(dim = d)

nodes_to_remove = [(0,5),(1,5),(3,5),(4,5),(5,5),(6,5),(7,5),(9,5),(10,5),(5,0),(5,1),(5,3),(5,4),(5,5),(5,6),(5,7),(5,9),(5,10)]
edges_to_remove = basegraph.edges(nodes_to_remove)
basegraph.remove_edges_from(copy.deepcopy(edges_to_remove))
# add_kleinberg_edges(g, n=1, r=2)


for i in range(10):
    g = copy.deepcopy(basegraph)
    add_random_edges(g, n=4)

    env = gym.make('graphworld-v0', graph = g, dim = d)
    nA = env.action_space.n

    Q = defaultdict(lambda: np.random.rand(nA))
    Q, lc = q_learning(env, n_episodes = 500, gamma = 0.95)
    plt.plot(lc)

plt.show()
