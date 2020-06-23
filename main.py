import numpy as np
import os
import util
import environment
import mc
import td
import lfa

cwd = os.getcwd()

def main():
    # Assignment 1: Implementation of Easy21.
    env = environment.Easy21()

    # Assignment 2: Monte-Carlo control in Easy21.
    model = mc.MonteCarlo(env)
    model.run(iterations=10000000)
    fig1 = util.plot_Vstar(model.Q)
    fig1.savefig(f'{cwd}/graphs/mc_vstar.png', bbox_inches='tight')

    # Assignment 3: TD learning in Easy21.
    model1 = td.Sarsa(env)
    lmbda_mse = {}
    for lmbda in list(np.arange(0, 11) / 10):
        model1.__init__(env)
        model1.run(lmbda, model.Q, iterations=10000)
        mse = np.sum(np.square(model1.Q - model.Q)) / 420
        print(f"MSE of Sarsa({lmbda}): ", mse)
        lmbda_mse[lmbda] = mse
        if lmbda == 0 or lmbda == 1:
            fig2 = util.plot_dct(model1.mse, 'Episode', 'MSE', f'Learning curve of lambda = {lmbda}')
            fig2.savefig(f'{cwd}/graphs/td_learnratelambda{lmbda}.png', bbox_inches='tight')
    fig3 = util.plot_dct(lmbda_mse, 'Lambda', 'MSE', 'MSE against Lambda')
    fig3.savefig(f'{cwd}/graphs/td_mselambda.png', bbox_inches='tight')

    # Assignment 4: Linear Function Approximation in Easy21.
    model2 = lfa.LinFuncApprox(env)
    lmbda_mse = {}
    for lmbda in list(np.arange(0, 11) / 10):
        model2.run(lmbda, model.Q, iterations=10000)
        mse = np.sum(np.square(model2.Q - model.Q)) / 420
        print(f"MSE of Sarsa({lmbda}) with a Linear Function Approximator: ", mse)
        lmbda_mse[lmbda] = mse
        if lmbda == 0 or lmbda == 1:
            fig4 = util.plot_dct(model2.mse, 'Episode', 'MSE', f'Learning curve of lambda = {lmbda}')
            fig4.savefig(f'{cwd}/graphs/lfa_learnratelambda{lmbda}.png', bbox_inches='tight')
    fig5 = util.plot_dct(lmbda_mse, 'Lambda', 'MSE', 'MSE against Lambda')
    fig5.savefig(f'{cwd}/graphs/lfa_mselambda.png', bbox_inches='tight')

if __name__ == '__main__':
    main()
