import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm

def mellowmax(values, omega = 1.0, axis = 0):
    n = values.shape[axis]
    return (np.log(np.sum(np.exp(omega * values))) - np.log(n)) / omega

def mellowmax_policy(Q, state, omega = 1.0):
    a = Q[state]
    prob = np.exp(a) / np.sum(np.exp(a))
    return np.random.choice(len(prob), p = prob)

def epsilon_greedy(Q, state, epsilon = 0.1):
    a = Q[state]
    if random.uniform(0,1) < epsilon:
        return random.randint(0, len(a)-1)
    return np.argmax(a)

def sarsa(env, n_episodes, gamma=0.95, alpha=0.01, epsilon=1.0):
    # OFF policy TD control
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for e in tqdm(range(n_episodes)):
        epsilon *= 0.99
        done = False
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        while not done:
            next_state, reward, done, info = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            td_target = reward + gamma * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            state = next_state
            action = next_action
    return Q

def sarsa_mm(env, n_episodes, gamma=0.95, alpha=0.01, omega=1.0):
    # OFF policy TD control
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for e in tqdm(range(n_episodes)):
        done = False
        state = env.reset()
        action = mellowmax_policy(Q, state, omega)
        while not done:
            next_state, reward, done, info = env.step(action)
            next_action = mellowmax_policy(Q, next_state, omega)

            td_target = reward + gamma * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            state = next_state
            action = next_action
    return Q

def q_learning(env, n_episodes, gamma=0.95, alpha=0.1, epsilon=1.0):
    # ON policy TD control
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    learning_curve = []
    q = 0
    for e in tqdm(range(n_episodes)):
        epsilon *= 0.99
        done = False
        state = env.reset()
        while not done:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, info = env.step(action)

            td_target = reward + gamma * np.amax(Q[next_state])
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            q += alpha * td_error
            state = next_state

        learning_curve.append(q)
    return Q, learning_curve

def q_learning_mm(env, n_episodes, gamma=0.95, alpha=0.01, omega=1.0):
    # ON policy TD control
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for e in tqdm(range(n_episodes)):
        done = False
        state = env.reset()
        while not done:
            action = mellowmax_policy(Q, state, omega)
            next_state, reward, done, info = env.step(action)

            td_target = reward + gamma * mellowmax(Q[next_state], omega)
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            state = next_state
    return Q
