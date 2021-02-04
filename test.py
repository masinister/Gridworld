import gym
import gym_graphworld
import networkx as nx
import copy
from collections import defaultdict
import numpy as np
from mc import epsilon_greedy, q_learning, sarsa
from utils import *
from basegraphs import *

d = (11,11)
g = fourrooms(*d)
add_random_edges(g, n=3)
# print(params(g))

env = gym.make('graphworld-v0', graph = g, dim = d)
nA = env.action_space.n
opt_Q = env.optimal_Q()

Q = defaultdict(lambda: np.random.rand(nA))
Q, lc = q_learning(env, n_episodes = 10000, gamma = 0.95, alpha = 0.1)
print(Q)
print(dict_error(opt_Q, Q))

for i in range(100):
    done = False
    state = env.reset()
    while not done:
        env.render()
        action = np.argmax(Q[state])
        next_state, reward, done, info = env.step(action)
        state = next_state
