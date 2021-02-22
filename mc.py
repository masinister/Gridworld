import numpy as np
import random
from collections import defaultdict, deque
from tqdm import tqdm

def boltzmann(Q, state, tau=0.5):
    a = Q[state]
    prob = np.exp(a / tau)
    prob /= np.sum(prob)
    return np.random.choice(len(prob), p = prob)

def epsilon_greedy(Q, state, epsilon = 0.5):
    a = Q[state]
    if random.uniform(0,1) < epsilon:
        return random.randint(0, len(a)-1)
    return np.argmax(a)

def q_learning(env, n_steps, gamma=0.95, alpha=0.05):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    learning_curve = []
    last30 = deque(maxlen = 30)
    q = 0
    step = 0
    while step < n_steps:
        done = False
        state = env.reset()
        while not done:
            step += 1
            action = epsilon_greedy(Q, state)
            next_state, reward, done, info = env.step(action)

            td_target = reward + gamma * np.amax(Q[next_state])
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            state = next_state

            q += alpha * td_error
            learning_curve.append(q)
        last30.append(q)
        error = np.abs(last30[-1] - last30[0])
        if len(last30) == 30 and error < 0.01:
            break
    return Q, learning_curve
