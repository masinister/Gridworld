import gym
import gym_graphworld
import networkx as nx
import copy
from collections import defaultdict
import ray
import time
import numpy as np
from mc import q_learning
from utils import *
from plot import plot
from basegraphs import *

ray.init(num_cpus = 6)

d = (11,11)
data = []
num_trials = 100
basegraphs = [fourrooms] * num_trials

@ray.remote
def run_one_trial(g):
    import gym_graphworld
    add_kleinberg_edges(g, n = np.random.randint(1, 10), r=1)

    env = gym.make('graphworld-v0', graph = g, dim = d)
    nA = env.action_space.n
    opt_Q = env.optimal_Q()
    target_Q = dict_error(opt_Q, defaultdict(lambda: np.zeros(5)))

    Q, lc = q_learning(env, n_steps = 1e5, target_Q = target_Q)
    return params(g,d), lc, dict_error(opt_Q, Q), dict_ratio(opt_Q, Q), target_Q

print("Starting experiment:")
start = time.time()
data = ray.get([run_one_trial.remote(basegraph(*d)) for basegraph in basegraphs])
end = time.time()
print("{} trials ran in {:.3f} seconds".format(num_trials,end - start))

np.save('data.npy', data)
plot("cover time")
