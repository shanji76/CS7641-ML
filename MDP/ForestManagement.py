
from hiive.mdptoolbox import mdp
import sys
import pandas as pd
import matplotlib.pyplot as plt
from hiive.examples import firemdp
from QLearner import QLearner


class ForestManagement():
    def __init__(self):
        self.name = 'forest'
        self.s=0.5
        self.prob, self.rewards = firemdp.getTransitionAndRewardArrays(self.s)

    def value_iteration(self, discount=0.999, epsilon=0.001, save_policy=False, save_plot=False):
        vi = mdp.ValueIteration(transitions=self.prob, reward=self.rewards, gamma=discount,epsilon=epsilon)

        run_stats = vi.run()
        self.plot(run_stats, 'Fire Management - Value Iteration')

        expected_values = vi.V
        optimal_policy = vi.policy
        iterations = vi.iter
        time = vi.time

        return [expected_values, optimal_policy, iterations, time]

    def policy_iteration(self, discount=0.999, save_policy=False, save_plot=False):
        pi = mdp.PolicyIteration(transitions=self.prob, reward=self.rewards, gamma=discount)
        run_stats = pi.run()
        self.plot(run_stats, 'Fire Management - Policy Iteration')

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
        plot_df['Rewards'] = [rs['Reward'] for rs in stats]
        plt.figure()
        plot_df.plot(x='Iterations', y='Error', title=title)
        plt.show()

        plt.figure()
        plot_df.plot(x='Time', y='Error', title=title)
        plt.show()

        plt.figure()
        plot_df.plot(x='Iterations', y='Rewards', title=title)
        plt.show()

    def move(self, action, state):
        x, F = firemdp.convertIndexToState(state)
        r = firemdp.getHabitatSuitability(F)
        F = self.transition_fire_state(F, action)
        x = firemdp.simulateTransition(x, self.s, r, F==0)
        state = firemdp.convertStateToIndex(x, F)
        if x == 0:
            reward = 0
            done = True
        else:
            reward = 1
            done = False
        return [state, reward, done]



    def transition_fire_state(self, F, a):
        """Transition the years since last fire based on the action taken.
        Parameters
        ----------
        F : int
            The time in years since last fire.
        a : int
            The action undertaken.
        Returns
        -------
        F : int
            The time in years since last fire.
        """
        ## Efect of action on time in years since fire.
        if a == 0:
            # Increase the time since the patch has been burned by one year.
            # The years since fire in patch is absorbed into the last class
            if F < firemdp.FIRE_CLASSES - 1:
                F += 1
        elif a == 1:
            # When the patch is burned set the years since fire to 0.
            F = 0

        return F

    def eval_policy(self, policy, iterations=100):
        rewards = []
        reward_iter = []
        for i in range(iterations):
            # reset environment
            state = firemdp.convertStateToIndex(1,0)
            # initialize
            steps = 0
            done = False
            iter_reward = 0
            while not done:
                action = policy[state]
                s_prime, reward, done = self.move(action, state)
                rewards.append(reward)
                iter_reward += reward
                state = s_prime
                if done:
                    break
                steps += 1
            reward_iter.append(iter_reward)
        print('Total reward={r} over {n} iterations'.format(r=sum(rewards), n=iterations))
        plt.figure()
        plt.plot(range(1,iterations+1), reward_iter)
        plt.show()


if __name__ == '__main__':
    SDP = firemdp.solveMDP()
    print("Finite Horizon")
    firemdp.printPolicy(SDP.policy[:, 0])

    fm = ForestManagement()
    vi, vi_policy, _, _ = fm.value_iteration()
    pi, pi_policy, _, _ = fm.policy_iteration()

    print("Value Iteration")
    firemdp.printPolicy(vi_policy)
    # fm.eval_policy(vi_policy,iterations=1000)
    print("Policy Iteration")
    firemdp.printPolicy(pi_policy)
    # fm.eval_policy(pi_policy, iterations=1000)

    ql = QLearner(fm.name, fm.prob, fm.rewards)
    ql.q_learning_trials(trials=20, vi=vi, pi=pi)
    run_stats,ql_policy = ql.q_learning(gamma=0.9911, alpha=0.3695, alpha_decay=0.9998, alpha_min=0.0747,
                                        epsilon=0.8608, epsilon_decay=0.9996, n_iter=47366,returnStats=True)
    print("QLearning Policy")
    firemdp.printPolicy(ql_policy)
    # fm.eval_policy(ql_policy, iterations=1000)

    sys.exit()