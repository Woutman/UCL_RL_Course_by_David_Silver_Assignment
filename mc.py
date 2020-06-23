import numpy as np
import random
import matplotlib.pyplot as plt


class MonteCarlo:
    def __init__(self, environment, N0=100):
        self.env = environment
        self.Q = np.zeros((10, 21, 2))
        self.Nv = np.zeros((10, 21))
        self.Nq = np.zeros((10, 21, 2))
        self.N0 = N0

    def epsilon_greedy(self, s):
        # Implementation of an epsilon greedy policy.
        epsilon = self.N0 / (self.N0 + self.Nv[s[0] - 1, s[1] - 1])
        choice = random.choices(['exploit', 'explore'], weights=[1 - epsilon, epsilon])
        if choice == 'exploit':
            a = np.argmax(self.Q[s[0] - 1, s[1] - 1, a] for a in [0, 1])
        else:
            a = random.randint(0, 1)
        return a

    def play(self):
        # Runs through one episode of the game.
        q_visited = []
        s = self.env.setup()
        while s != 'terminal':
            self.Nv[s[0] - 1][s[1] - 1] += 1
            a = self.epsilon_greedy(s)
            r, s1 = self.env.step(s, a)
            q_visited.append((s + (a,) + (r,)))
            s = s1
        return q_visited

    def run(self, iterations):
        # Runs through the specified number of episodes and updates the state-action function accordingly.
        for i in range(iterations):
            q_visited = self.play()
            g = sum(q[3] for q in q_visited)
            for q in q_visited:
                self.Nq[q[0] - 1, q[1] - 1, q[2]] += 1
                alpha = 1 / self.Nq[q[0] - 1, q[1] - 1, q[2]]
                self.Q[q[0] - 1, q[1] - 1, q[2]] += alpha * (g - self.Q[q[0] - 1, q[1] - 1, q[2]])

    def plot_Vstar(self):
        # Plots the optimized state value function.
        fig = plt.figure(figsize=[15, 8])
        ax = fig.add_subplot(111, projection='3d')

        x = range(10)
        y = range(21)
        X, Y = np.meshgrid(x, y)

        V_star = np.zeros((21, 10))
        for i in range(len(self.Q)):
            for j in range(len(self.Q[i])):
                V_star[j, i] = max(self.Q[i, j])
        Z = np.array(V_star)

        ax.set_title("V* for each state")
        ax.set_xlabel('Dealer showing')
        ax.set_ylabel('Player total')
        ax.set_zlabel('V*')
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
        plt.show()
