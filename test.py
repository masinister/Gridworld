import gym
import gym_graphworld
import networkx as nx
import copy
from collections import defaultdict
import numpy as np
from mc import epsilon_greedy, q_learning
from utils import *
from basegraphs import *
from heuristics import *

d = (11,11)
g = fourrooms(*d)
add_random_edges(g, n=10)
# print(params(g, d))
print(num_shortest_paths(g, t = (10,10)))


env = gym.make('graphworld-v0', graph = g, dim = d, noise = 0.1)
nA = env.action_space.n
opt_Q = env.optimal_Q()
target = np.sum([np.sum(np.abs(opt_Q[k])) for k in opt_Q.keys()])

Q = defaultdict(lambda: np.random.rand(nA))
Q, lc = q_learning(env, n_steps = 5e4, target_Q = target)
print("error:", dict_error(opt_Q, Q), dict_ratio(opt_Q, Q))

plt.figure(1)
plt.plot(lc)
plt.show()

for i in range(100):
    done = False
    state = env.reset()
    while not done:
        env.render()
        action = np.argmax(Q[state])
        next_state, reward, done, info = env.step(action)
        state = next_state
