import time
import mlrose
from mlrose import random_hill_climb, DiscreteOpt, FourPeaks, Queens, \
    FlipFlop, simulated_annealing, genetic_alg, mimic
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Utility import extractData
import matplotlib.pyplot as plt
from sklearn import decomposition, preprocessing, metrics
import os.path


class Problem():
    def __init__(self, verbose=False):
        self.verbose = verbose

    def rhc(self, max_attempts=10, max_iters=500, restarts=100):
        start = time.time()
        best_state, best_fitness, curve = random_hill_climb(self.problem_fit,
                                                            max_attempts=max_attempts,
                                                            max_iters=max_iters,
                                                            restarts=restarts,
                                                            curve=True,
                                                            random_state=111)
        end = time.time()
        time_elapsed = end - start
        return [best_fitness, time_elapsed, curve]

    def sim_annealing(self, max_attempts=10, max_iters=np.inf, decay='geom'):
        decay_lookup = {
            'geom': mlrose.GeomDecay(),
            'exp': mlrose.ExpDecay(),
            'arith': mlrose.ArithDecay()
        }
        start = time.time()
        best_state, best_fitness = simulated_annealing(self.problem_fit, max_attempts=max_attempts,
                                                       max_iters=max_iters,
                                                       schedule=decay_lookup[decay],
                                                       random_state=111)
        end = time.time()
        time_elapsed = end - start
        return [best_fitness, time_elapsed]

    def genetic(self, max_attempts=10, max_iters=np.inf, pop_size=10, mutation_prob=0.1):
        start = time.time()
        best_state, best_fitness = genetic_alg(self.problem_fit, max_attempts=max_attempts,
                                               max_iters=max_iters,
                                               pop_size=pop_size,
                                               mutation_prob=mutation_prob,
                                               random_state=111)
        end = time.time()
        time_elapsed = end - start
        return [best_fitness, time_elapsed]

    def mimic(self, max_attempts=10, max_iters=np.inf, pop_size=10, keep_pct=0.5):
        start = time.time()
        best_state, best_fitness = mimic(self.problem_fit, max_attempts=max_attempts,
                                         max_iters=max_iters,
                                         pop_size=pop_size,
                                         keep_pct=keep_pct,
                                         random_state=111)
        end = time.time()
        time_elapsed = end - start
        return [best_fitness, time_elapsed]

    def test_random_hill(self, trials=10, max_attempts_range=[10, 50, 100],
                         random_restarts_range=[100],
                         iterations=[200]):
        data = []
        print("Random Hill Climbing")
        file = 'data/{p}_rhc.csv'.format(p=self.problem)

        for i in range(5, iterations, 5):
            for m in max_attempts_range:
                for r in random_restarts_range:
                    best_fitness, time_elapsed, curve = self.rhc(max_attempts=m, max_iters=i, restarts=r)
                    row = [i, m, r, best_fitness, time_elapsed]
                    if self.verbose:
                        print("Iterations: {i}, Max-Attempts: {m}, Restarts:{r}, Fitness: {f}, T={t}".format(i=i, r=r,
                                                                                                             m=m,
                                                                                                             f=best_fitness,
                                                                                                             t=round(
                                                                                                                 time_elapsed,
                                                                                                                 2)))
                    data.append(row)
                    df = pd.DataFrame(data, columns=['iterations', 'max_attempts', 'restarts', 'best_fitness', 'time'])
                    fn = file
                    df.to_csv(fn, index=False)

        self.plotAlgorithmPerformce(file, 'RHC')

    def test_sim_annealing(self, max_attempts_range=[50, 100],
                           iterations=[200], decay_range=['geom']):
        data = []
        print("Simulated Annealing")
        file = 'data/{p}_sa.csv'.format(p=self.problem)

        for i in range(5, iterations, 5):
            for m in max_attempts_range:
                for d in decay_range:
                    best_fitness, time_elapsed = self.sim_annealing(max_attempts=m, max_iters=i, decay=d)
                    row = [i, m, d, best_fitness, time_elapsed]
                    if self.verbose:
                        print(
                            "Iterations: {i}, Max-Attempts: {m}, Decay: {d}, Fitness: {f}, T={t}".format(i=i, m=m, d=d,
                                                                                                         f=best_fitness,
                                                                                                         t=round(
                                                                                                             time_elapsed,
                                                                                                             2)))
                    data.append(row)
                    df = pd.DataFrame(data, columns=['iterations', 'max_attempts', 'schedule', 'best_fitness', 'time'])
                    fn = file
                    df.to_csv(fn, index=False)

        self.plotAlgorithmPerformce(file, 'Simulated Annealing')

    def test_genetic(self, max_attempts_range=[50, 100],
                     iterations=[200], pop_range=[100], mutation_range=[0.1]):
        data = []
        print("Genetic Algorithm")
        file = 'data/{p}_ga.csv'.format(p=self.problem)

        for i in range(5, iterations, 5):
            for m in max_attempts_range:
                for p in pop_range:
                    for mp in mutation_range:
                        best_fitness, time_elapsed = self.genetic(max_attempts=m, max_iters=i, pop_size=p,
                                                                  mutation_prob=mp)
                        row = [i, m, p, mp, best_fitness, time_elapsed]
                        if self.verbose:
                            print(
                                "Iterations: {i}, Max-Attempts: {m}, Pop-Size: {p},Keep-Pct:{mp}, Fitness: {f}, T={t}".format
                                (i=i, m=m, p=p, mp=mp, f=best_fitness, t=round(time_elapsed, 2)))
                        data.append(row)
                        df = pd.DataFrame(data,
                                          columns=['iterations', 'max_attempts', 'pop', 'mutation', 'best_fitness',
                                                   'time'])
                        fn = file
                        df.to_csv(fn, index=False)

        self.plotAlgorithmPerformce(file, 'Genetic Algorithm')

    def test_mimic(self, max_attempts_range=[50, 100],
                   iterations=[200], pop_range=[100], keep_pct_range=[0.1]):
        data = []
        print("MIMIC Algorithm")
        file = 'data/{p}_mim.csv'.format(p=self.problem)

        for i in range(5, iterations, 5):
            for m in max_attempts_range:
                for p in pop_range:
                    for kp in keep_pct_range:
                        best_fitness, time_elapsed = self.mimic(max_attempts=m, max_iters=i, pop_size=p,
                                                                keep_pct=kp)
                        row = [i, m, p, kp, best_fitness, time_elapsed]
                        if self.verbose:
                            print(
                                "Iterations: {i}, Max-Attempts: {m}, Pop-Size: {p}, Keep-Pct:{kp}, Fitness: {f}, T={t}".format
                                (i=i, m=m, p=p, kp=kp, f=best_fitness, t=round(time_elapsed, 2)))
                        data.append(row)
                        df = pd.DataFrame(data,
                                          columns=['iterations', 'max_attempts', 'pop', 'keep-pct', 'best_fitness',
                                                   'time'])
                        fn = file
                        df.to_csv(fn, index=False)

        self.plotAlgorithmPerformce(file, 'MIMIC Algorithm')


class NeuralNetwork(Problem):
    def __init__(self, verbose=False):
        self.problem = 'nn'
        self.name = 'Neural Network'
        self.verbose = False
        self.data_prep()
        np.seterr(over='ignore')

    def data_prep(self, test_size=0.3):
        x, y = extractData("data/default_of_credit_card_clients.csv")
        self.num_classes = len(np.unique(y))
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, test_size=test_size)

    def train(self, algorithm='random_hill_climb', restarts=0, pop_size=200,
              mutation_prob=0.1, max_iters_range=[200], learning_rate=0.001,
              layers=8, activation='relu',
              early_stopping=False):
        for max_iters in range(10,max_iters_range,10):
            nn = mlrose.NeuralNetwork(hidden_nodes=layers,
                                      activation=activation,
                                      algorithm=algorithm,
                                      restarts=restarts,  # rhc
                                      pop_size=pop_size,  # ga
                                      mutation_prob=mutation_prob,  # ga
                                      max_iters=max_iters, bias=False,
                                      is_classifier=True,
                                      learning_rate=learning_rate,
                                      early_stopping=early_stopping,
                                      max_attempts=1000, curve=True)

            # scale data
            train_x = self.train_x
            test_x = self.test_x
            train_y = self.train_y
            test_y = self.test_y

            # fit model
            start = time.time()
            nn.fit(train_x, train_y)
            end = time.time()
            time_elapsed = end - start
            iterations = len(nn.fitness_curve)
            train_y_pred = nn.predict(train_x)
            test_y_pred = nn.predict(test_x)
            # accuracy
            train_error = metrics.accuracy_score(train_y, train_y_pred)
            test_error = metrics.accuracy_score(test_y, test_y_pred)
            # write to df
            data = [[algorithm, restarts, pop_size, mutation_prob, max_iters,
                     learning_rate, layers, activation, early_stopping, train_error,
                     test_error,
                     time_elapsed, iterations, nn.fitness_curve.mean()]]
            cols = ['algorithm', 'restarts', 'pop_size', 'mutation_prob', 'max_iters',
                    'learning_rate', 'layers', 'activation', 'early_stopping',
                    'train_error', 'test_error',
                    'time_elapsed', 'iterations','fitness']
            df = pd.DataFrame(data, columns=cols)
            for c, d in zip(cols, data[0]):
                print("{c}: {d}".format(c=c, d=d))
            # write to file
            filename = 'data/nn_results_'+algorithm+'.csv'
            if os.path.isfile(filename):
                df_ = pd.read_csv(filename)
                df = df_.append(df)
            df.to_csv(filename, index=False)

    def tune(self, algorithm='random_hill_climb', restarts_range=[0], pop_size_range=[200],
                  mutation_prob_range=[0.1], max_iters_range=[200], learning_rate_range=[0.001],
                  layers_range=[[8]], activation_range=['relu'],
                  early_stopping=False):
            for restarts in restarts_range:
                for pop_size in pop_size_range:
                    for mutation_prob in mutation_prob_range:
                        for max_iters in max_iters_range:
                            for learning_rate in learning_rate_range:
                                for layers in layers_range:
                                    for activation in activation_range:
                                        nn = mlrose.NeuralNetwork(hidden_nodes=layers,
                                                                  activation=activation,
                                                                  algorithm=algorithm,
                                                                  restarts=restarts,  # rhc
                                                                  pop_size=pop_size,  # ga
                                                                  mutation_prob=mutation_prob,  # ga
                                                                  max_iters=max_iters, bias=False,
                                                                  is_classifier=True,
                                                                  learning_rate=learning_rate,
                                                                  early_stopping=early_stopping,
                                                                  max_attempts=1000, curve=True)

                                        train_x = self.train_x
                                        test_x = self.test_x
                                        train_y = self.train_y
                                        test_y = self.test_y

                                        # fit model
                                        start = time.time()
                                        nn.fit(train_x, train_y)
                                        end = time.time()
                                        time_elapsed = end - start
                                        iterations = len(nn.fitness_curve)
                                        train_y_pred = nn.predict(train_x)
                                        test_y_pred = nn.predict(test_x)
                                        # accuracy
                                        train_error = metrics.accuracy_score(train_y, train_y_pred)
                                        test_error = metrics.accuracy_score(test_y, test_y_pred)
                                        # write to df
                                        data = [[algorithm, restarts, pop_size, mutation_prob, max_iters,
                                                 learning_rate, layers, activation, early_stopping, train_error,
                                                 test_error,
                                                 time_elapsed, iterations, nn.fitness_curve.mean()]]
                                        cols = ['algorithm', 'restarts', 'pop_size', 'mutation_prob', 'max_iters',
                                                'learning_rate', 'layers', 'activation', 'early_stopping',
                                                'train_error', 'test_error',
                                                'time_elapsed', 'iterations','fitness']
                                        df = pd.DataFrame(data, columns=cols)
                                        for c, d in zip(cols, data[0]):
                                            print("{c}: {d}".format(c=c, d=d))
                                        # write to file
                                        filename = 'data/nn_results_'+algorithm+'_tune.csv'
                                        if os.path.isfile(filename):
                                            df_ = pd.read_csv(filename)
                                            df = df_.append(df)
                                        df.to_csv(filename, index=False)

def plotAlgorithmPerformce(self, csvFile, algo):
    rhc_data = pd.read_csv(csvFile)
    max_fitness = rhc_data["best_fitness"].max()
    print("Plotting")
    img = 'plots/' + csvFile[5:-4]
    rhc_grp = rhc_data.groupby(by=["iterations"])

    plt.figure()
    plt.plot(rhc_grp.groups.keys(),
             rhc_grp.agg({'best_fitness': 'mean'}))
    plt.xlabel("Iterations")
    plt.ylabel("Best Fitness")
    plt.title(self.name + ' ' + algo)
    plt.savefig(img + '_iter.png')

    plt.figure()
    plt.plot(rhc_grp.agg({'time': 'mean'}),
             rhc_grp.agg({'best_fitness': 'mean'}))
    plt.xlabel("Time Taken")
    plt.ylabel("Best Fitness")
    plt.title(self.name + ' ' + algo)
    plt.savefig(img + '_time.png')


class FourPeaksProblem(Problem):
    def __init__(self, length=10, t_pct=0.1, verbose=False):
        self.problem = 'fourpeaks{l}'.format(l=length)
        self.verbose = verbose
        self.name = 'Four Peaks'
        fitness_fn = FourPeaks(t_pct=t_pct)
        self.problem_fit = DiscreteOpt(length=length, fitness_fn=fitness_fn,
                                       maximize=True, max_val=2)


class QueensProblem(Problem):
    def __init__(self, length=8, verbose=False):
        self.problem = 'queens{l}'.format(l=length)
        self.verbose = verbose
        self.name = 'N Queens'
        fitness_fn = Queens()
        self.problem_fit = DiscreteOpt(length=length, fitness_fn=fitness_fn,
                                       maximize=True, max_val=length)


class FlipFlopProblem(Problem):
    def __init__(self, length=8, verbose=False):
        self.problem = 'flipflop{l}'.format(l=length)
        self.verbose = verbose
        self.name = 'Flip Flop'
        fitness_fn = FlipFlop()
        self.problem_fit = DiscreteOpt(length=length, fitness_fn=fitness_fn,
                                       maximize=True)


def main():
    random_restarts_range = [50, 100]  # rhc
    iterations = 250
    decay_range = ['geom', 'exp', 'arith']

    print("FOUR PEAKS")
    fp = FourPeaksProblem(length=40, t_pct=0.1, verbose=True)
    fp.test_random_hill(trials=2, max_attempts_range=[100], random_restarts_range=random_restarts_range,
                        iterations=iterations)
    fp = FourPeaksProblem(length=50, t_pct=0.1, verbose=True)
    fp.test_sim_annealing(max_attempts_range=[150], iterations=iterations, decay_range=decay_range)
    fp.test_genetic(max_attempts_range=[100], iterations=iterations, pop_range=[100, 200],
                    mutation_range=[0.05, 0.1, 0.2])
    fp.test_mimic(max_attempts_range=[50], iterations=50, pop_range=[100, 200],
                  keep_pct_range=[0.1, 0.15])

    print("N Queens")
    qp = QueensProblem(length=12, verbose=True)
    qp.test_random_hill(trials=1, max_attempts_range=[100], random_restarts_range=random_restarts_range,
                        iterations=iterations)
    qp.test_sim_annealing(max_attempts_range=[150], iterations=iterations, decay_range=decay_range)
    qp.test_genetic(max_attempts_range=[100], iterations=iterations, pop_range=[100, 200],
                    mutation_range=[0.05, 0.1, 0.2])
    qp.test_mimic(max_attempts_range=[100, 120], iterations=iterations, pop_range=[200],
                  keep_pct_range=[0.15, 0.2, 0.25])

    print("Flip-Flop")
    fl = FlipFlopProblem(length=60, verbose=True)
    fl.test_random_hill(trials=2, max_attempts_range=[100], random_restarts_range=random_restarts_range,
                        iterations=iterations)
    fl.test_sim_annealing(max_attempts_range=[150], iterations=iterations, decay_range=decay_range)
    fl.test_genetic(max_attempts_range=[100], iterations=iterations, pop_range=[100, 200],
                    mutation_range=[0.05, 0.1, 0.2])
    fl.test_mimic(max_attempts_range=[100], iterations=50, pop_range=[100, 200],
                  keep_pct_range=[0.1, 0.15])

    # NEURAL NET
    print("Neural Networks")
    # tune hyper parameters
    nn = NeuralNetwork(verbose=True)
    nn.tune('gradient_descent', learning_rate_range=[0.0005, 0.001, 0.005], max_iters_range=[500, 1000, 2000],
            layers_range=[[15]], activation_range=['relu'])
    nn.tune('random_hill_climb', learning_rate_range=[0.75, 1.0], restarts_range=[0,50,100],
             max_iters_range=[10000, 25000, 50000],
             layers_range=[[16]], activation_range=['relu'])
    nn.tune('simulated_annealing', learning_rate_range=[0.75, 1.0], max_iters_range=[50000, 750000],
             layers_range=[[16]], activation_range=['relu'])
    nn.tune('genetic_alg', learning_rate_range=[0.75, 1.0], pop_size_range=[10000, 20000, 30000],
             mutation_prob_range=[0.01, 0.1],
             max_iters_range=[50, 100, 500], layers_range=[[16]], activation_range=['relu'])

    # train/test plots
    nn = NeuralNetwork(verbose=True)
    nn.train('gradient_descent', learning_rate=0.05, max_iters_range=200,
             layers=[15], activation='relu')
    nn.train('random_hill_climb', learning_rate=0.05, max_iters_range=200,
             layers=[10], activation='relu')
    nn.train('simulated_annealing', learning_rate=0.05, max_iters_range=200,
             layers=[10], activation='relu')
    nn.train('genetic_alg', learning_rate=0.05, max_iters_range=200,
             layers=[10], activation='relu')


if __name__ == "__main__":
    main()
