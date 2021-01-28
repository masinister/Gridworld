import gym
import sys
import os
import time
import copy
import numpy as np
from random import sample
import matplotlib.pyplot as plt
from gym import error, spaces, utils
from gym.utils import seeding
from PIL import Image as Image

# define colors
# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red
COLORS = {0:[0.0,0.0,0.0], 1:[0.5,0.5,0.5], \
          2:[0.0,0.0,1.0], 3:[0.0,1.0,0.0], \
          4:[1.0,0.0,0.0], 6:[1.0,0.0,1.0], \
          7:[1.0,1.0,0.0]}

class GraphworldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, graph, dim):
        super(GraphworldEnv, self).__init__()
        self.actions = [0, 1, 2, 3, 4]
        self.action_space = spaces.Discrete(5)
        self.action_pos_dict = {0: [0,0], 1:[-1, 0], 2:[1,0], 3:[0,-1], 4:[0,1]}

        self.dim = dim
        self.obs_shape = [self.dim[0], self.dim[1], 3]
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float32)

        self.graph = copy.deepcopy(graph)
        self.walls = [node for node in self.graph.nodes() if self.graph.degree(node) == 0]
        self.floors = [node for node in self.graph.nodes() if node not in self.walls]
        self.teleporters = [node for node in self.graph.nodes() if self._teleport_edge(node)]

        self.agent_start_state = (0,0)
        self.agent_target_state = (self.dim[0] - 1, self.dim[1] - 1)
        self.agent_state = copy.deepcopy(self.agent_start_state)

        self.observation = self._next_observation()

    def step(self, action):
        info = {}
        info['success'] = False
        if self._take_action(action):
            self.observation = self._next_observation()
            info['success'] = True
        if self.agent_state == self.agent_target_state:
            return self.observation.tobytes(), 1.0, True, info
        return self.observation.tobytes(), 0.0, False, info

    def reset(self):
        # self.agent_state = sample(self.floors, 1)[0]
        self.agent_state = self.agent_start_state
        self.observation = self._next_observation()
        return self.observation.tobytes()

    def render(self, mode='human', close=False):
        img = self.observation
        fig = plt.figure(0)
        plt.clf()
        plt.imshow(img)
        fig.canvas.draw()
        plt.pause(0.000001)

    def _take_action(self, action):
        if action == 0 and self.agent_state in self.teleporters:
            edge = self._teleport_edge(self.agent_state)
            self.agent_state = edge[1]
            return True

        next_state = (self.agent_state[0] + self.action_pos_dict[action][0],
                      self.agent_state[1] + self.action_pos_dict[action][1])
        if (self.agent_state, next_state) in self.graph.edges():
            self.agent_state = next_state
            return True
        return False

    def _next_observation(self):
        obs = np.zeros(self.obs_shape)
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                obs[i,j] = np.array(COLORS[0])
                if (i,j) in self.walls:
                    obs[i,j] = np.array(COLORS[1])
                elif (i,j) == self.agent_target_state:
                    obs[i,j] = np.array(COLORS[2])
                elif (i,j) == self.agent_state:
                    obs[i,j] = np.array(COLORS[3])
                elif (i,j) in self.teleporters:
                    obs[i,j] = np.array(COLORS[4])
        return obs

    def _teleport_edge(self, node):
        grid_edges = [(node, node2) for node2 in [(node[0]+1, node[1]),(node[0]-1, node[1]),(node[0], node[1]+1),(node[0], node[1]-1)]]
        non_grid_edges = [e for e in self.graph.edges(node) if e not in grid_edges]
        return non_grid_edges[0] if non_grid_edges != [] else False

    def close(self):
        plt.close(0)
        return
