import gym
import gym_graphworld
import networkx as nx
import copy
from collections import defaultdict
import numpy as np
from mc import q_learning, sarsa
from utils import add_kleinberg_edges, add_random_edges, params
from plot import plot
import concurrent.futures as cf

d = (11,11)
basegraph = nx.grid_graph(dim = d)

nodes_to_remove = [(0,5),(1,5),(3,5),(4,5),(5,5),(6,5),(7,5),(9,5),(10,5),(5,0),(5,1),(5,3),(5,4),(5,5),(5,6),(5,7),(5,9),(5,10)]
edges_to_remove = basegraph.edges(nodes_to_remove)
basegraph.remove_edges_from(copy.deepcopy(edges_to_remove))

def run_one_trial(g):
    add_random_edges(g, n = np.random.randint(1, 5))

    env = gym.make('graphworld-v0', graph = g, dim = d)
    nA = env.action_space.n

    Q, lc = q_learning(env, n_episodes = 500, gamma = 0.95)
    return params(g), [lc]

data = []
num_trials = 2

with cf.ThreadPoolExecutor(max_workers=1) as executor:
    futures = [executor.submit(run_one_trial, copy.deepcopy(basegraph)) for _ in range(num_trials)]
    for trial in cf.as_completed(futures):
        data.append(trial.result())

np.save('data.npy', data)
plot()
