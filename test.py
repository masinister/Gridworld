import gym
import gym_graphworld
import networkx as nx
import copy
from collections import defaultdict
import numpy as np
from mc import epsilon_greedy, q_learning, sarsa
from utils import add_kleinberg_edges, add_random_edges, params
from basegraphs import fourrooms

d = (11,11)
g = fourrooms(*d)
add_random_edges(g, n=1)
print(params(g))

env = gym.make('graphworld-v0', graph = g, dim = d)
nA = env.action_space.n

Q = defaultdict(lambda: np.random.rand(nA))
Q, lc = q_learning(env, n_episodes = 500, gamma = 0.95)

for i in range(100):
    done = False
    state = env.reset()
    while not done:
        env.render()
        action = np.argmax(Q[state])
        next_state, reward, done, info = env.step(action)
        state = next_state
