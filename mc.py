import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    if random.uniform(0,1) < epsilon:
        return random.randint(0, nA-1)
    return np.argmax(Q[state])

def sarsa(env, n_episodes, gamma=0.95, alpha=0.01, epsilon=1.0):
    # OFF policy TD control
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for e in tqdm(range(n_episodes)):
        epsilon *= 0.99
        done = False
        state = env.reset()
        action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
        while not done:
            next_state, reward, done, info = env.step(action)
            next_action = epsilon_greedy(Q, next_state, env.action_space.n, epsilon)

            td_target = reward + gamma * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            state = next_state
            action = next_action
    return Q

def q_learning(env, n_episodes, gamma=0.95, alpha=0.01, epsilon=1.0):
    # ON policy TD control
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for e in tqdm(range(n_episodes)):
        epsilon *= 0.99
        done = False
        state = env.reset()
        while not done:
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            next_state, reward, done, info = env.step(action)

            td_target = reward + gamma * np.max(Q[next_state])
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            state = next_state
    return Q
