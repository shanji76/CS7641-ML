from mdptoolbox.example import forest
from mdptoolbox import mdp
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ForestManagement():
    def __init__(self):
        self.name = 'forest'
        self.render = False

    def value_iteration(self, S=10, rw=10, rc=5, p=0.9):
        P,R = forest(S=S,r1=rw, r2=rc, p=p)
        vi = mdp.ValueIteration(transitions=P, reward=R,discount=p)

        vi.run()
        # self.plot(run_stats,'Frozen Lake - Value Iteration')

        expected_values = vi.V
        optimal_policy = vi.policy
        iterations = vi.iter
        time = vi.time

        return [expected_values, optimal_policy, iterations, time]

    def policy_iteration(self, S=10, rw=10, rc=5, p=0.9):
        P, R = forest(S=S, r1=rw, r2=rc, p=p)
        pi = mdp.PolicyIteration(transitions=P, reward=R, discount=p)
        pi.run()
        # self.plot(run_stats,'Frozen Lake - Policy Iteration')

        expected_values = pi.V
        optimal_policy = pi.policy
        iterations = pi.iter
        time = pi.time

        return [expected_values, optimal_policy, iterations, time]

    def plot(self, stats, title):
        plot_df = pd.DataFrame()
        plot_df['Iterations'] = range(1, len(stats) + 1)
        plot_df['Error'] = [rs['Error'] for rs in stats]
        plot_df['Time'] = [rs['Time'] for rs in stats]
        plt.figure()
        plot_df.plot(x='Iterations', y='Error', title=title)
        plt.show()

        plt.figure()
        plot_df.plot(x='Time', y='Error', title=title)
        plt.show()

    def eval_policy(self, policy, iterations=100):

        print('Total reward={r} over {n} iterations'.format(r=4, n=iterations))

if __name__ == '__main__':
    fm = ForestManagement()
    vi, vi_policy, _,_ = fm.value_iteration(S=2000)
    pi, pi_policy, _, _ = fm.policy_iteration(S=2000)

    sys.exit()