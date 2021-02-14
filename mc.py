import numpy as np
import random
from collections import defaultdict, deque
from tqdm import tqdm

def mellowmax(Q, state, omega = 10.0):
    a = Q[state]
    prob = np.exp(a) / np.sum(np.exp(a))
    return np.random.choice(len(prob), p = prob)

def epsilon_greedy(Q, state, epsilon = 0.1):
    a = Q[state]
    if random.uniform(0,1) < epsilon:
        return random.randint(0, len(a)-1)
    return np.argmax(a)

def q_learning(env, n_episodes, gamma=0.95, alpha=0.1, epsilon=1.0):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    learning_curve = []
    last30 = deque(maxlen = 30)
    q = 0
    for e in range(n_episodes):
        epsilon = max(epsilon * 0.99, 0.1)
        done = False
        state = env.reset()
        while not done:
            action = mellowmax(Q, state)
            next_state, reward, done, info = env.step(action)

            td_target = reward + gamma * np.amax(Q[next_state])
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            state = next_state

            q += alpha * td_error
        learning_curve.append(q)
        last30.append(q)
        if len(last30) == 30 and max(last30) < 1e-4:
            break
    return Q, learning_curve
