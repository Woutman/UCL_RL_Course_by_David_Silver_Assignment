import numpy as np
import random


class LinFuncApprox:
    def __init__(self, environment):
        self.env = environment
        self.w = np.zeros(36)
        self.Q = np.zeros((10, 21, 2))
        self.mse = {}
        self.epsilon = 0.05
        self.alpha = 0.01
        # Create list of features.
        self.f = []
        d = list(tuple(zip(range(1, 8, 3), range(4, 11, 3))))
        p = list(tuple(zip(range(1, 17, 3), range(6, 22, 3))))
        a = [(0,), (1,)]
        for i in range(len(d)):
            for j in range(len(p)):
                for k in range(len(a)):
                    temp = [d[i], p[j], a[k]]
                    self.f.append(temp)

    def sa_to_x(self, s, a):
        # Converts state-action pair to feature vector.
        x = np.zeros(36)
        for idx, feat in enumerate(self.f):
            x[idx] = (feat[0][0] <= s[0] <= feat[0][1] and feat[1][0] <= s[1] <= feat[1][1] and a == feat[2][0])
        return x

    def sa_to_q(self, s, a):
        # Calculates state-action value from state-action pair.
        x = self.sa_to_x(s, a)
        return sum(x * self.w)

    def epsilon_greedy(self, s):
        # Implementation of an epsilon greedy policy.
        choice = random.choices(['exploit', 'explore'], weights=[1 - self.epsilon, self.epsilon])
        if choice == 'exploit':
            a = np.argmax(self.sa_to_q(s, a) for a in [0, 1])
        else:
            a = random.randint(0, 1)
        return a

    def play(self, lmbda):
        # Runs through one episode of the game and updates the feature weights after each step
        # according to eligibility traces.
        E = np.zeros(36)
        s = self.env.setup()
        a = self.epsilon_greedy(s)
        while s != 'terminal':
            r, s1 = self.env.step(s, a)
            if s1 == 'terminal':
                td_error = r - self.sa_to_q(s, a)
            else:
                a1 = self.epsilon_greedy(s1)
                td_error = r + self.sa_to_q(s1, a1) - self.sa_to_q(s, a)
                a = a1
            E = lmbda * E + self.sa_to_x(s, a)
            self.w += self.alpha * td_error * E
            s = s1

    def form_Q(self):
        # Forms a 3D matrix containing all state-action values.
        for d in range(1, 11):
            for p in range(1, 22):
                for a in range(2):
                    s = (d, p)
                    self.Q[d - 1, p - 1, a] = self.sa_to_q(s, a)

    def run(self, lmbda, Q_true, iterations):
        # Initializes random weights and runs through the specified number of episodes.
        # Also stores learning rate for lambda = 0 and lambda = 1.
        self.w = np.random.rand(36)
        self.Q = np.zeros((10, 21, 2))
        self.mse = {}
        for i in range(iterations):
            self.play(lmbda)
            if lmbda == 0 or lmbda == 1:
                if i % 100 == 0:
                    self.form_Q()
                    mse = np.sum(np.square(self.Q - Q_true)) / 420
                    self.mse[i] = mse
        self.form_Q()
