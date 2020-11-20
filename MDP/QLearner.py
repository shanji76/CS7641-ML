import pandas as pd
import numpy as np
from hiive.mdptoolbox import mdp
import matplotlib.pyplot as plt


class QLearner():
    def __init__(self, name, prob, rewards):
        self.name = name
        self.prob = prob
        self.rewards = rewards

    def q_learning_trials(self, trials=10, vi=None, pi=None):
        for t in range(trials):
            np.random.seed()  # remove seed for parameter generation
            n_iter = np.random.randint(50000, 100000)
            gamma = np.random.uniform(0.45, 1.0)
            alpha = np.random.uniform(0.2, 1.0)
            alpha_decay = 1 - np.exp(np.random.uniform(np.log(0.0001), np.log(0.001)))
            alpha_min = np.random.uniform(0, 0.3)
            epsilon = np.random.uniform(0.5, 1.0)
            epsilon_decay = 1 - np.exp(np.random.uniform(np.log(0.0001), np.log(0.001)))
            epsilon_min = np.random.uniform(0, 0.9)


            results = self.q_learning(gamma=gamma, alpha=alpha, alpha_decay=alpha_decay,
                                  alpha_min=alpha_min, epsilon=epsilon,
                                  epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
                                  n_iter=n_iter)
            row = [t, results[2], gamma, alpha, alpha_decay, alpha_min, epsilon, epsilon_decay,
                   epsilon_min]
            row.append(results[3])  # timed
            row.append(np.sum(np.absolute(np.array(vi) - np.array(pi))))
            row.append(np.sum(np.absolute(results[0] - np.array(vi))))
            row.append(np.sum(np.absolute(results[0] - np.array(pi))))
            row.append(results[4]) #Mean-V

            try:
                df = pd.read_csv('{0}_qlearning_trials.csv'.format(self.name))
                df.loc[len(df)] = row
                df.to_csv('{0}_qlearning_trials.csv'.format(self.name), index=False)
            except FileNotFoundError:
                cols = ['trial', 'n_iter', 'gamma', 'alpha', 'alpha_decay', 'alpha_min',
                        'epsilon', 'epsilon_decay', 'epsilon_min', 'time', 'vi_pi_error', 'vi_error', 'pi_error','mean_v']
                df = pd.DataFrame([row], columns=cols)
                df.to_csv('{0}_qlearning_trials.csv'.format(self.name), index=False)

        self.plot(self.name+' - QLearning')

    def q_learning(self, gamma=0.9, alpha=0.1, alpha_decay=0.99, alpha_min=0.1, epsilon=1.0,
                   epsilon_min=0.1, epsilon_decay=0.99, n_iter=10000, returnStats=False):
        ql = mdp.QLearning(self.prob, self.rewards, gamma,
                           alpha=alpha, alpha_decay=alpha_decay, alpha_min=alpha_min,
                           epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
                           n_iter=n_iter)
        run_stats = ql.run()
        # self.plot(run_stats, 'Frozen Lake - Q-Learning')
        expected_values = ql.V
        optimal_policy = ql.policy
        time = ql.time
        if(not returnStats):
            return [expected_values, optimal_policy, len(run_stats), time,np.sum([rs['Mean V'] for rs in run_stats])]
        return run_stats, optimal_policy

    def plot(self, title):
        plot_df = pd.read_csv('{0}_qlearning_trials.csv'.format(self.name))
        plot_df= plot_df.sort_values(by=['gamma'])
        plt.figure()
        plot_df.plot(y='n_iter', x='gamma', title=title)
        plt.show()
