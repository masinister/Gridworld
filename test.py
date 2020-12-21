import gym
import gym_graphworld
import networkx as nx
import copy
from collections import defaultdict
import numpy as np
from mc import epsilon_greedy, q_learning, q_learning_mm, sarsa, sarsa_mm
from utils import add_kleinberg_edges, add_random_edges

d = (11,11)
g = nx.grid_graph(dim = d)

nodes_to_remove = [(0,5),(1,5),(3,5),(4,5),(5,5),(6,5),(7,5),(9,5),(10,5),(5,0),(5,1),(5,3),(5,4),(5,5),(5,6),(5,7),(5,9),(5,10)]
edges_to_remove = g.edges(nodes_to_remove)
g.remove_edges_from(copy.deepcopy(edges_to_remove))
# add_kleinberg_edges(g, n=1, r=2)
add_random_edges(g, n=1)

env = gym.make('graphworld-v0', graph = g, dim = d)
nA = env.action_space.n

Q = defaultdict(lambda: np.random.rand(nA))
Q = sarsa_mm(env, n_episodes = 100, gamma = 0.95)

for i in range(100):
    done = False
    state = env.reset()
    while not done:
        env.render()
        action = np.argmax(Q[state])
        next_state, reward, done, info = env.step(action)
        state = next_state
