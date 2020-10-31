from Clustering import Clustering
from Decomposing import Decomposing
from Utility import extractData
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
from sys import exit
from sklearn import preprocessing

def main():

    exp_results = pd.DataFrame(columns=['Data Set','Dim. Red. Algo','Cluster Algo.','# Clusters','Mutual Info Score'])
    data_file = "data/winequality-red.csv"
    x, y = extractData(data_file)
    x = pd.DataFrame(preprocessing.scale(x), columns=x.columns)
    dimreduce = Decomposing()
    cluster = Clustering()

    pca_result = pd.DataFrame(dimreduce.pca_eval(x,y,'experiment/wine',5,'Rating'))
    km = cluster.kmeansFitBestModel(1, 2, 'experiment/wine', pca_result.copy(), 4)
    score = adjusted_mutual_info_score(km.labels_,y)
    print('Wine : PCA - Kmeans : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['Wine Quality','PCA','k-Means',4,score])
    em = cluster.emFitBestModel(1,2,'experiment/wine',5,pca_result)
    score = adjusted_mutual_info_score(em.predict(pca_result), y)
    print('Wine : PCA - EM : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['Wine Quality', 'PCA', 'Exp. Maximization', 5, score])

    ica_result = pd.DataFrame(dimreduce.ica_eval(x, y, 'experiment/wine', 11, 'Rating'))
    km = cluster.kmeansFitBestModel(1, 2, 'experiment/wine', ica_result.copy(), 5)
    score = adjusted_mutual_info_score(km.labels_, y)
    print('Wine : ICA - Kmeans : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['Wine Quality', 'ICA', 'k-Means', 5, score])
    em = cluster.emFitBestModel(1, 2, 'experiment/wine', 5, ica_result)
    score = adjusted_mutual_info_score(em.predict(ica_result), y)
    print('Wine : ICA - EM : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['Wine Quality', 'ICA', 'Exp. Maximization', 5, score])

    rp_result = pd.DataFrame(dimreduce.rp_eval(x, y, 'experiment/wine', 11, 'Rating'))
    km = cluster.kmeansFitBestModel(1, 2, 'experiment/wine', rp_result.copy(), 5)
    score = adjusted_mutual_info_score(km.labels_, y)
    print('Wine : RP - Kmeans : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['Wine Quality', 'RP', 'k-Means', 5, score])
    em = cluster.emFitBestModel(1, 2, 'experiment/wine', 5, rp_result)
    score = adjusted_mutual_info_score(em.predict(rp_result), y)
    print('Wine : RP - EM : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['Wine Quality', 'RP', 'Exp. Maximization', 5, score])

    sk_result = pd.DataFrame(dimreduce.sk_eval(x, y, 8))
    km = cluster.kmeansFitBestModel(1, 2, 'experiment/wine', sk_result.copy(), 4)
    score = adjusted_mutual_info_score(km.labels_, y)
    print('Wine : SK - Kmeans : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['Wine Quality', 'Select-K', 'k-Means', 4, score])
    em = cluster.emFitBestModel(1, 2, 'experiment/wine', 5, sk_result)
    score = adjusted_mutual_info_score(em.predict(sk_result), y)
    print('Wine : SK - EM : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['Wine Quality', 'Select-K', 'Exp. Maximization', 5, score])


    data_file = "data/default_of_credit_card_clients.csv"
    x, y = extractData(data_file)
    x = pd.DataFrame(preprocessing.scale(x), columns=x.columns)

    pca_result = pd.DataFrame(dimreduce.pca_eval(x, y, 'experiment/default', 10,'Defaulted'))
    km = cluster.kmeansFitBestModel(1, 2, 'experiment/default', pca_result, 5)
    score = adjusted_mutual_info_score(km.labels_, y)
    print('Default : PCA - Kmeans : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['CC Default', 'PCA', 'k-Means', 5, score])
    em = cluster.emFitBestModel(1, 2, 'experiment/wine', 5, pca_result)
    score = adjusted_mutual_info_score(em.predict(pca_result), y)
    print('Default : PCA - EM : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['CC Default', 'PCA', 'Exp. Maximization', 5, score])

    ica_result = pd.DataFrame(dimreduce.ica_eval(x, y, 'experiment/default', 6, 'Defaulted'))
    km = cluster.kmeansFitBestModel(1, 2, 'experiment/default', ica_result, 4)
    score = adjusted_mutual_info_score(km.labels_, y)
    print('Default : ICA - Kmeans : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['CC Default', 'ICA', 'k-Means', 6, score])
    em = cluster.emFitBestModel(1, 2, 'experiment/wine', 4, ica_result)
    score = adjusted_mutual_info_score(em.predict(ica_result), y)
    print('Default : ICA - EM : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['CC Default', 'ICA', 'Exp. Maximization', 4, score])

    rp_result = pd.DataFrame(dimreduce.rp_eval(x, y, 'experiment/default', 24, 'Defaulted'))
    km = cluster.kmeansFitBestModel(1, 2, 'experiment/default', rp_result, 10)
    score = adjusted_mutual_info_score(km.labels_, y)
    print('Default : RP - Kmeans : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['CC Default', 'RP', 'k-Means', 10, score])
    em = cluster.emFitBestModel(1, 2, 'experiment/wine', 10, rp_result)
    score = adjusted_mutual_info_score(em.predict(rp_result), y)
    print('Default : RP - EM : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['CC Default', 'RP', 'Exp. Maximization', 10, score])

    sk_result = pd.DataFrame(dimreduce.sk_eval(x, y, 6))
    km = cluster.kmeansFitBestModel(1, 2, 'experiment/default', sk_result, 4)
    score = adjusted_mutual_info_score(km.labels_, y)
    print('Default : SK - Kmeans : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['CC Default', 'Select-K', 'k-Means', 4, score])
    em = cluster.emFitBestModel(1, 2, 'experiment/wine', 4, rp_result)
    score = adjusted_mutual_info_score(em.predict(rp_result), y)
    print('Default : SK - EM : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['CC Default', 'Select-K', 'Exp. Maximization', 4, score])

    print(exp_results)
    print(exp_results.to_latex())


def append_results(exp_results, results):
    return exp_results.append(
        {'Data Set': results[0], 'Dim. Red. Algo': results[1], 'Cluster Algo.': results[2], '# Clusters': results[3],
         'Mutual Info Score': results[4]},
        ignore_index=True)


if __name__ == "__main__":
    main()
    exit()
