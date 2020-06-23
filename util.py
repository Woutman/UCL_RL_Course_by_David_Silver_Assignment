import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_dct(dct, x_val, y_val, name):
    # Helper function for plotting a 2D linegraph from a dictionary.
    fig = plt.figure(figsize=[15, 8])
    ax = plt.axes()
    x = [key for key in dct]
    y = [dct[key] for key in x]
    ax.plot(x, y)
    ax.set(xlabel=x_val, ylabel=y_val, title=name)
    plt.show()
    return fig

def plot_Vstar(Q):
    # Plots the optimized state-value function.
    fig = plt.figure(figsize=[15, 8])
    ax = fig.add_subplot(111, projection='3d')

    x = range(10)
    y = range(21)
    X, Y = np.meshgrid(x, y)

    V_star = np.zeros((21, 10))
    for i in range(len(Q)):
        for j in range(len(Q[i])):
            V_star[j, i] = max(Q[i, j])
    Z = np.array(V_star)

    ax.set_title("V* for each state")
    ax.set_xlabel('Dealer showing')
    ax.set_ylabel('Player total')
    ax.set_zlabel('V*')
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    plt.show()

    return fig
