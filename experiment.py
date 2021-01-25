import gym
import gym_graphworld
import networkx as nx
import copy
from collections import defaultdict
import ray
import time
import numpy as np
from mc import q_learning, sarsa
from utils import add_kleinberg_edges, add_random_edges, params
from plot import plot
from basegraphs import fourrooms

ray.init()

d = (11,11)
basegraph = fourrooms(*d)

@ray.remote
def run_one_trial(g):
    import gym_graphworld
    add_random_edges(g, n = np.random.randint(1, 5))

    env = gym.make('graphworld-v0', graph = g, dim = d)
    nA = env.action_space.n

    Q, lc = q_learning(env, n_episodes = 500, gamma = 0.95)
    return params(g), [lc]

data = []
num_trials = 10
print("Starting experiment:")

start = time.time()
data = ray.get([run_one_trial.remote(copy.deepcopy(basegraph)) for i in range(num_trials)])
end = time.time()
print("{} trials ran in {:.3f} seconds".format(num_trials,end - start))

np.save('data.npy', data)
plot()
