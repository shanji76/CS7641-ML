from QLearner import QLearner
from hiive.mdptoolbox import mdp
import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv

class FrozenLake():
    def __init__(self, size=10, p=0.8):
        self.name='frozenlake'
        self.size=size
        random_map = generate_random_map(size=size,p=p)
        self.env = gym.make("FrozenLake-v0", desc=random_map, is_slippery=True)
        self.env.seed(123)
        self.env.action_space.np_random.seed(123)
        self.env._max_episode_steps=20000
        self.prob = self.probability_matrix()
        self.rewards = self.rewards_matrix()
        self.env.render()

    def probability_matrix(self):
        prob = np.zeros((self.env.nA, self.env.nS, self.env.nS))
        print(prob.shape)
        for state,transitions in self.env.P.items():
            for action in range(self.env.nA):
                li = transitions[action]
                for state_info in li:
                    new_state_prob = state_info[0]
                    new_state = state_info[1]
                    prob[action][state][new_state] += new_state_prob
        return prob

    def rewards_matrix(self):
        rewards = np.zeros((self.env.nA, self.env.nS, self.env.nS))
        for state, transitions in self.env.P.items():
            for action in range(self.env.nA):
                li = transitions[action]
                for state_info in li:
                    new_state = state_info[1]
                    reward = state_info[2]
                    rewards[action][state][new_state] = reward
        return rewards

    def value_iteration(self, discount=0.999, epsilon=0.001, save_policy=False, save_plot=False):
        vi = mdp.ValueIteration(transitions=self.prob, reward=self.rewards,gamma=discount,epsilon=epsilon)

        run_stats = vi.run()
        self.plot(run_stats,'Frozen Lake - Value Iteration')

        expected_values = vi.V
        optimal_policy = vi.policy
        iterations = vi.iter
        time = vi.time

        return [expected_values, optimal_policy, iterations, time]

    def policy_iteration(self, discount=0.999, save_policy=False, save_plot=False):
        pi = mdp.PolicyIteration(transitions=self.prob, reward=self.rewards, gamma=discount)
        run_stats = pi.run()
        self.plot(run_stats,'Frozen Lake - Policy Iteration')

        expected_values = pi.V
        optimal_policy = pi.policy
        iterations = pi.iter
        time = pi.time

        return [expected_values, optimal_policy, iterations, time]

    def eval_policy(self, policy, iterations=100):
        rewards = []
        reward_iter = []
        for i in range(iterations):
            # reset environment
            state = self.env.reset()
            steps = 0
            done = False
            iter_reward = 0
            while not done:
                # self.env.render()
                action = policy[state]
                s_prime, reward, done, info = self.env.step(action)
                state = s_prime
                t_reward = self.tweak_reward(reward,done)
                rewards.append(reward)
                iter_reward += reward
                if done:
                    # print("episode: {}/{}, score: {:.2f}, steps: {}".format(i, iterations, iter_reward, steps))
                    break
                steps += 1


            reward_iter.append(iter_reward)
        plt.figure()
        plt.plot(range(1,iterations+1), reward_iter)
        plt.show()

        print('Total reward={r} over {n} iterations'.format(r=sum(rewards), n=iterations))

    def tweak_reward(self, reward, done):
        if reward == 0:
            reward = 0.001
        if done:
            if reward < 1:
                reward = -1
        return reward

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

    def plotQlearn(self, stats, title):
        plot_df = pd.DataFrame()
        plot_df['Iterations'] = range(1, len(stats) + 1)
        plot_df['Mean-V'] = [rs['Mean V'] for rs in stats]

        plt.figure()
        plot_df.plot(x='Iterations', y='Mean-V', title=title)
        plt.show()



if __name__ == '__main__':
    fl = FrozenLake(size=20)
    vi, vi_policy, it, t = fl.value_iteration()
    pi, pi_policy, _, _ = fl.policy_iteration(discount=0.9)

    print('Eval : Value Iteration Policy')
    fl.eval_policy(vi_policy, iterations=1000)
    print('Eval : Policy Iteration Policy')
    fl.eval_policy(pi_policy, iterations=1000)

    ql = QLearner(fl.name, fl.prob, fl.rewards)
    ql.q_learning_trials(trials=20,vi=vi,pi=pi)
    run_stats,ql_policy = ql.q_learning(gamma=0.999,alpha=0.45,alpha_decay=0.999907088,alpha_min=0.082682664,
                                       epsilon=0.968587999,epsilon_min=0.148113184, epsilon_decay=0.996218224, n_iter=60000, returnStats=True)
    fl.plotQlearn(run_stats, 'Frozen Lake - Q-Learning')
    # fl.eval_policy(ql_policy,iterations=1000)

    sys.exit()




    




