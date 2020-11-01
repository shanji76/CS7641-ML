from sys import exit

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
from Clustering import Clustering
from Decomposing import Decomposing
from Utility import extractData


def nn_classifier(train_x, train_y, test_x, test_y):
    nn = MLPClassifier(hidden_layer_sizes=(5,),activation='logistic',solver='adam', max_iter=500)
    model = nn.fit(train_x, train_y)
    pred_y =  model.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, pred_y) * 100
    print('Accuracy = {}'.format(accuracy))
    return accuracy

def plot_results(title, name, x_vals, y_vals, x_label, y_label):
    plt.figure()
    plt.bar(x_vals, y_vals)
    plt.xticks(range(len(x_vals)), x_vals)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig('image/experiment/'+name+'.png')

def main():

    exp_results = pd.DataFrame(columns=['Data Set','Dim. Red. Algo','Cluster Algo.','# Clusters','Mutual Info Score'])
    data_file = "data/winequality-red.csv"
    x, y = extractData(data_file)
    x = pd.DataFrame(preprocessing.scale(x), columns=x.columns)
    dimreduce = Decomposing()
    cluster = Clustering()

    pca_result = pd.DataFrame(dimreduce.pca_eval(x,y,'experiment/wine/pca',5,'Rating')[0])
    km = cluster.kmeansFitBestModel(1, 2, 'experiment/wine/pca', pca_result.copy(), 5)
    score = adjusted_mutual_info_score(km.labels_,y)
    print('Wine : PCA - Kmeans : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['Wine Quality','PCA','k-Means',5,score])
    em = cluster.emFitBestModel(1,2,'experiment/wine/pca',5,pca_result)
    score = adjusted_mutual_info_score(em.predict(pca_result), y)
    print('Wine : PCA - EM : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['Wine Quality', 'PCA', 'Exp. Maximization', 5, score])

    ica_result = pd.DataFrame(dimreduce.ica_eval(x, y, 'experiment/wine/ica', 11, 'Rating')[0])
    km = cluster.kmeansFitBestModel(1, 2, 'experiment/wine/ica', ica_result.copy(), 5)
    score = adjusted_mutual_info_score(km.labels_, y)
    print('Wine : ICA - Kmeans : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['Wine Quality', 'ICA', 'k-Means', 5, score])
    em = cluster.emFitBestModel(1, 2, 'experiment/wine/ica', 5, ica_result)
    score = adjusted_mutual_info_score(em.predict(ica_result), y)
    print('Wine : ICA - EM : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['Wine Quality', 'ICA', 'Exp. Maximization', 5, score])

    rp_result = pd.DataFrame(dimreduce.rp_eval(x, y, 'experiment/wine/rp', 11, 'Rating')[0])
    km = cluster.kmeansFitBestModel(1, 2, 'experiment/wine/rp', rp_result.copy(), 5)
    score = adjusted_mutual_info_score(km.labels_, y)
    print('Wine : RP - Kmeans : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['Wine Quality', 'RP', 'k-Means', 5, score])
    em = cluster.emFitBestModel(1, 2, 'experiment/wine/rp', 5, rp_result)
    score = adjusted_mutual_info_score(em.predict(rp_result), y)
    print('Wine : RP - EM : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['Wine Quality', 'RP', 'Exp. Maximization', 5, score])

    sk_result = pd.DataFrame(dimreduce.sk_eval(x, y, 8)[0])
    km = cluster.kmeansFitBestModel(1, 2, 'experiment/wine/sk', sk_result.copy(), 4)
    score = adjusted_mutual_info_score(km.labels_, y)
    print('Wine : SK - Kmeans : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['Wine Quality', 'Select-K', 'k-Means', 4, score])
    em = cluster.emFitBestModel(1, 2, 'experiment/wine/sk', 5, sk_result)
    score = adjusted_mutual_info_score(em.predict(sk_result), y)
    print('Wine : SK - EM : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['Wine Quality', 'Select-K', 'Exp. Maximization', 5, score])


    data_file = "data/default_of_credit_card_clients.csv"
    x, y = extractData(data_file)
    x = pd.DataFrame(preprocessing.scale(x), columns=x.columns)

    pca_result = pd.DataFrame(dimreduce.pca_eval(x, y, 'experiment/default/pca', 10,'Defaulted')[0])
    km = cluster.kmeansFitBestModel(1, 2, 'experiment/default/pca', pca_result, 5)
    score = adjusted_mutual_info_score(km.labels_, y)
    print('Default : PCA - Kmeans : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['CC Default', 'PCA', 'k-Means', 5, score])
    em = cluster.emFitBestModel(1, 2, 'experiment/default/pca', 5, pca_result)
    score = adjusted_mutual_info_score(em.predict(pca_result), y)
    print('Default : PCA - EM : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['CC Default', 'PCA', 'Exp. Maximization', 5, score])

    ica_result = pd.DataFrame(dimreduce.ica_eval(x, y, 'experiment/default/ica', 6, 'Defaulted')[0])
    km = cluster.kmeansFitBestModel(1, 2, 'experiment/default/ica', ica_result, 4)
    score = adjusted_mutual_info_score(km.labels_, y)
    print('Default : ICA - Kmeans : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['CC Default', 'ICA', 'k-Means', 6, score])
    em = cluster.emFitBestModel(1, 2, 'experiment/default/ica', 4, ica_result)
    score = adjusted_mutual_info_score(em.predict(ica_result), y)
    print('Default : ICA - EM : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['CC Default', 'ICA', 'Exp. Maximization', 4, score])

    rp_result = pd.DataFrame(dimreduce.rp_eval(x, y, 'experiment/default/rp', 24, 'Defaulted')[0])
    km = cluster.kmeansFitBestModel(1, 2, 'experiment/default/rp', rp_result, 10)
    score = adjusted_mutual_info_score(km.labels_, y)
    print('Default : RP - Kmeans : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['CC Default', 'RP', 'k-Means', 10, score])
    em = cluster.emFitBestModel(1, 2, 'experiment/default/rp', 10, rp_result)
    score = adjusted_mutual_info_score(em.predict(rp_result), y)
    print('Default : RP - EM : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['CC Default', 'RP', 'Exp. Maximization', 10, score])

    sk_result = pd.DataFrame(dimreduce.sk_eval(x, y, 6)[0])
    km = cluster.kmeansFitBestModel(1, 2, 'experiment/default/sk', sk_result, 4)
    score = adjusted_mutual_info_score(km.labels_, y)
    print('Default : SK - Kmeans : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['CC Default', 'Select-K', 'k-Means', 4, score])
    em = cluster.emFitBestModel(1, 2, 'experiment/default/sk', 4, rp_result)
    score = adjusted_mutual_info_score(em.predict(rp_result), y)
    print('Default : SK - EM : Score = {}'.format(score))
    exp_results = append_results(exp_results, ['CC Default', 'Select-K', 'Exp. Maximization', 4, score])

    print(exp_results)
    print(exp_results.to_latex())

    ##### NN
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)

    accuracy_raw = nn_classifier(train_x, train_y, test_x, test_y)

    pca_result, pca = dimreduce.pca_eval(train_x, train_y, 'nn', 10, 'Defaulted')
    test_data = pca.transform(test_x)
    accuracy_pca = nn_classifier(pca_result, train_y, test_data, test_y)

    ica_result, ica = dimreduce.ica_eval(train_x, train_y, 'nn', 6, 'Defaulted')
    test_data = ica.transform(test_x)
    accuracy_ica = nn_classifier(ica_result, train_y, test_data, test_y)

    rp_result, rp = dimreduce.rp_eval(train_x, train_y, 'nn', 24, 'Defaulted')
    test_data = rp.transform(test_x)
    accuracy_rp = nn_classifier(rp_result, train_y, test_data, test_y)

    sk_result, sk = dimreduce.sk_eval(train_x, train_y, 4)
    test_data = sk.transform(test_x)
    accuracy_sk = nn_classifier(sk_result, train_y, test_data, test_y)

    plot_results("Dimension Reduction Algo. Accuracy",'dim_red_accuracy',
                 ['Original','PCA','ICA','RP','SK'],
                 [accuracy_raw,accuracy_pca,accuracy_ica,accuracy_rp,accuracy_sk],'Dim. Red. Algorithms','Accuracy')

    km  = cluster.kmeansFitBestModel(1,2,'nn',train_x,10)
    km_data = km.fit_transform(train_x)
    test_data = km.transform(test_x)
    accuracy_km = nn_classifier(km_data,train_y,test_data, test_y)

    em = cluster.emFitBestModel(1, 2, 'nn', 15,train_x)
    em_train_labels = em.predict(train_x)
    em_train_ohc = one_hot_encode(em_train_labels, 15)
    em_train = np.concatenate((train_x, em_train_ohc), 1)
    # one hot encode cluster labels to val set
    em_test_labels = em.predict(test_x)
    em_test_ohc = one_hot_encode(em_test_labels, 15)
    em_test = np.concatenate((test_x, em_test_ohc), 1)
    # scale data
    scaler = preprocessing.StandardScaler().fit(em_train)
    em_data = scaler.transform(em_train)
    test_data = scaler.transform(em_test)

    accuracy_em = nn_classifier(em_data, train_y, test_data, test_y)

    plot_results("Clustering Accuracy", 'cluster_accuracy',['Original', 'k-Means', 'Ex. Maximization'],
                 [accuracy_raw, accuracy_km, accuracy_em], 'Clustering Algorithms',
                 'Accuracy')

def one_hot_encode(labels, num_clusters=12):
        ohc = np.zeros((labels.shape[0], num_clusters))
        ohc[np.arange(labels.shape[0]), labels] = 1
        return ohc
def append_results(exp_results, results):
    return exp_results.append(
        {'Data Set': results[0], 'Dim. Red. Algo': results[1], 'Cluster Algo.': results[2], '# Clusters': results[3],
         'Mutual Info Score': results[4]},
        ignore_index=True)


if __name__ == "__main__":
    main()
    exit()
