from hiive.mdptoolbox import mdp
import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from gym.envs.toy_text.frozen_lake import generate_random_map

random_map = generate_random_map(size=20, p=0.8)

class FrozenLake():
    def __init__(self, size=10, p=0.8):
        self.name='frozenlake'
        self.size=size
        random_map = generate_random_map(size=size,p=p)
        self.env = gym.make("FrozenLake-v0", desc=random_map)
        self.env.seed(123)
        self.env.action_space.np_random.seed(123)
        self.env._max_episode_steps=20000
        self.prob = self.probability_matrix()
        self.rewards = self.rewards_matrix()

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
                rewards.append(t_reward)
                iter_reward += t_reward
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
            reward = -0.001
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

    def q_learning_trials(self, trials=10, vi=None, pi=None):
        for t in range(trials):
            np.random.seed()  # remove seed for parameter generation
            if self.name == 'frozenlake':
                n_iter = np.random.randint(10000, 50000)
                gamma = np.random.uniform(0.95, 1.0)
                alpha = np.random.uniform(0.2, 1.0)
                alpha_decay = 1 - np.exp(np.random.uniform(np.log(0.0001), np.log(0.001)))
                alpha_min = np.random.uniform(0, 0.1)
                epsilon = np.random.uniform(0.5, 1.0)
                epsilon_decay = 1 - np.exp(np.random.uniform(np.log(0.0001), np.log(0.001)))
                epsilon_min = np.random.uniform(0, 0.2)
            np.random.seed(1234)  # reset seed
            row = [t, n_iter, gamma, alpha, alpha_decay, alpha_min, epsilon, epsilon_decay,
                   epsilon_min]
            results = self.q_learning(gamma=gamma, alpha=alpha, alpha_decay=alpha_decay,
                                      alpha_min=alpha_min, epsilon=epsilon,
                                      epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
                                      n_iter=n_iter)
            row.append(results[3])  # timed
            row.append(np.sum(np.absolute(np.array(vi) - np.array(pi))))
            row.append(np.sum(np.absolute(results[0] - np.array(vi))))
            row.append(np.sum(np.absolute(results[0] - np.array(pi))))

            try:
                df = pd.read_csv('{0}_qlearning_trials.csv'.format(self.name))
                df.loc[len(df)] = row
                df.to_csv('{0}_qlearning_trials.csv'.format(self.name), index=False)
            except FileNotFoundError:
                cols = ['trial', 'n_iter', 'gamma', 'alpha', 'alpha_decay', 'alpha_min',
                        'epsilon', 'epsilon_decay', 'epsilon_min', 'time', 'vi_pi_error', 'vi_error', 'pi_error']
                df = pd.DataFrame([row], columns=cols)
                df.to_csv('{0}_qlearning_trials.csv'.format(self.name), index=False)

    def q_learning(self, gamma=0.9, alpha=0.1, alpha_decay=0.99, alpha_min=0.1, epsilon=1.0,
                   epsilon_min=0.1, epsilon_decay=0.99, n_iter=10000):
        ql = mdp.QLearning(self.prob, self.rewards, gamma,
                           alpha=alpha, alpha_decay=alpha_decay, alpha_min=alpha_min,
                           epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
                           n_iter=n_iter)
        run_stats = ql.run()
        # self.plot(run_stats, 'Frozen Lake - Q-Learning')
        expected_values = ql.V
        optimal_policy = ql.policy
        time = ql.time
        return [expected_values, optimal_policy, n_iter, time]

if __name__ == '__main__':
    fl = FrozenLake(size=20)
    vi, vi_policy, it, t = fl.value_iteration()
    pi, pi_policy, _, _ = fl.policy_iteration(discount=0.9)

    print('Eval : Value Iteration Policy')
    fl.eval_policy(vi_policy, iterations=1000)
    print('Eval : Policy Iteration Policy')
    fl.eval_policy(pi_policy, iterations=1000)

    fl.q_learning_trials(trials=20,vi=vi,pi=pi)
    ql, ql_policy, _,_ = fl.q_learning(gamma=0.9598,alpha=0.9464,alpha_decay=0.9998,alpha_min=0.07962,
                                       epsilon=0.9172,epsilon_decay=0.9998,n_iter=19000)
    fl.eval_policy(ql_policy,iterations=1000)

    sys.exit()




    




