import numpy as np
import random
from collections import defaultdict, deque
from tqdm import tqdm

def boltzmann(Q, state, tau=0.5):
    a = Q[state]
    prob = np.exp(a / tau)
    prob /= np.sum(prob)
    return np.random.choice(len(prob), p = prob)

def epsilon_greedy(Q, state, epsilon = 0.95):
    a = Q[state]
    if random.uniform(0,1) < epsilon:
        return random.randint(0, len(a)-1)
    return np.argmax(a)

def q_learning(env, n_episodes, gamma=0.95, alpha=0.1, epsilon=0.5):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    learning_curve = []
    last30 = deque(maxlen = 30)
    q = 0
    for e in range(n_episodes):
        done = False
        state = env.reset()
        while not done:
            action = boltzmann(Q, state)
            next_state, reward, done, info = env.step(action)

            td_target = reward + gamma * np.amax(Q[next_state])
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            state = next_state

            q += alpha * td_error
        learning_curve.append(q)
        last30.append(q)
        if len(last30) == 30 and np.std(last30) < 0.01:
            break
    return Q, learning_curve
