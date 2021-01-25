import gym
import gym_graphworld
import networkx as nx
import copy
from collections import defaultdict
import concurrent.futures as cf
import numpy as np
from mc import q_learning, sarsa
from utils import add_kleinberg_edges, add_random_edges, params
from plot import plot
from basegraphs import fourrooms

d = (11,11)
basegraph = fourrooms(*d)

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
