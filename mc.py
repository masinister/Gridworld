import numpy as np
import random
from collections import defaultdict, deque
from tqdm import tqdm

def boltzmann(Q, state, tau=0.5):
    a = Q[state]
    prob = np.exp(a / tau)
    prob /= np.sum(prob)
    return np.random.choice(len(prob), p = prob)

def epsilon_greedy(Q, state, epsilon = 0.25):
    a = Q[state]
    if random.uniform(0,1) < epsilon:
        return random.randint(0, len(a)-1)
    return np.argmax(a)

def q_learning(env, n_steps, gamma=0.95, alpha=0.25, target_Q = 1, tol = 0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    learning_curve = []
    tail = deque(maxlen = 1000)
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
            if step % 100 == 0:
                learning_curve.append(q / target_Q)
                tail.append(q)
        if len(tail) == 1000 and np.abs(tail[-1] - tail[0]) < tol:
            break
    return Q, learning_curve
