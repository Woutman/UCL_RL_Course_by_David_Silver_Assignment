import numpy as np
import random


class Sarsa:
    def __init__(self, environment, N0=100):
        self.env = environment
        self.Q = np.zeros((10, 21, 2))
        self.E = np.zeros((10, 21, 2))
        self.Nv = np.zeros((10, 21))
        self.Nq = np.zeros((10, 21, 2))
        self.N0 = N0
        self.mse = {}

    def epsilon_greedy(self, s):
        # Implementation of an epsilon greedy policy.
        epsilon = self.N0 / (self.N0 + self.Nv[s[0] - 1, s[1] - 1])
        choice = random.choices(['exploit', 'explore'], weights=[1 - epsilon, epsilon])
        if choice == 'exploit':
            a = np.argmax(self.Q[s[0] - 1, s[1] - 1, a] for a in [0, 1])
        else:
            a = random.randint(0, 1)
        return a

    def alpha(self, q):
        # Calculates learning rate
        return 1 / self.Nq[q[0], q[1], q[2]]

    def play(self, lmbda):
        # Runs through one episode of the game and updates the state value function after each step.
        q_visited = []
        s = self.env.setup()
        a = self.epsilon_greedy(s)
        while s != 'terminal':
            self.E[s[0] - 1, s[1] - 1, a] += 1
            r, s1 = self.env.step(s, a)
            q = s + (a,)
            q_visited.append(q)
            self.Nq[q[0] - 1, q[1] - 1, q[2]] += 1
            if s1 == 'terminal':
                td_error = r - self.Q[s[0] - 1, s[1] - 1, a]
                s = s1
            else:
                a1 = self.epsilon_greedy(s1)
                td_error = r + self.Q[s1[0] - 1, s1[1] - 1, a1] - self.Q[s[0] - 1, s[1] - 1, a]
                s, a = s1, a1
            for (i, j, k) in set(q_visited):
                self.Q[i - 1, j - 1, k] += self.alpha((i - 1, j - 1, k)) * td_error * self.E[i - 1, j - 1, k]
                self.E[i - 1, j - 1, k] *= lmbda

    def run(self, lmbda, Q_true, iterations):
        # Runs through the specified number of episodes and stores learning rate for lambda = 0 and lambda = 1.
        for i in range(iterations):
            self.E = np.zeros((10, 21, 2))
            self.play(lmbda)
            if lmbda == 0 or lmbda == 1:
                mse = np.sum(np.square(self.Q - Q_true)) / 420
                self.mse[i] = mse
