import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pandas import Series


class Analysis :

    def compare(self, algo, name):
        problems = {'Four Peaks':'fourpeaks50','N-Queens':'queens12','FlipFlop':'flipflop60'}
        plt.figure()
        for p in problems.keys():
            file = problems.get(p)+'_'+algo+'.csv'
            results = pd.read_csv('data/'+file)
            grp = results.groupby(by=["iterations"])
            plt.plot(grp.groups.keys(),
                     grp.agg({'best_fitness': 'mean'}),label=p)

        plt.xlabel("Iterations")
        plt.ylabel("Best Fitness")
        plt.title(name)
        plt.legend()
        plt.savefig("plots/"+algo+"_comp_iter.png")

        plt.figure()
        for p in problems.keys():
            file = problems.get(p) + '_' + algo + '.csv'
            results = pd.read_csv('data/' + file)
            grp = results.groupby(by=["iterations"])
            plt.plot(grp.agg({'time': 'mean'}),
                     grp.agg({'best_fitness': 'mean'}), label=p)

        plt.xlabel("Time Taken")
        plt.ylabel("Best Fitness")
        plt.title(name)
        plt.legend()
        plt.savefig("plots/"+algo+"_comp_time.png")

    def analyzeNN(self):
        gd_results = pd.read_csv('data/nn_results_gradient_descent.csv')
        rhc_results = pd.read_csv('data/nn_results_random_hill_climb.csv')
        sa_results = pd.read_csv('data/nn_results_simulated_annealing.csv')
        ga_results = pd.read_csv('data/nn_results_genetic_alg.csv')

        plt.figure()
        plt.plot(gd_results["iterations"], gd_results["fitness"], label="Gradient Descent")
        plt.plot(rhc_results["iterations"], rhc_results["fitness"], label="RHC")
        plt.plot(sa_results["iterations"], sa_results["fitness"], label="SA")
        plt.plot(ga_results["iterations"], ga_results["fitness"], label="GA")
        plt.xlabel("Iterations")
        plt.ylabel("Best Fitness")
        plt.legend()
        plt.savefig("plots/nn_comp_iter.png")

def main():
    analysis = Analysis()
    analysis.compare('rhc','Random Hill Climbing')
    analysis.compare('sa', 'Simulated Annealing')
    analysis.compare('ga', 'Genetic Algorithm')
    analysis.compare('mim', 'MIMIC Algorithm')

    analysis.analyzeNN()

if __name__ == '__main__':
        main()