from sklearn.cluster import KMeans
from Utility import extractData
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from collections import Counter
import pandas as pd
from sys import exit

class Clustering:

    def kMeansCluster(self, dataTitle, x, y, col1, col2):
        inertia = []
        kmeans_accuracy = {}
        k_range = [1,2,5,10,15]
        for k in k_range :
            km = KMeans(n_clusters=k,
                             n_init=10,
                             random_state=123)
            km.fit(x)
            inertia.append(km.inertia_)
            kmeans_accuracy[k]=self.cluster_accuracy(y,km.predict(x))
        plt.figure()
        plt.plot(k_range, inertia, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.savefig('image/'+dataTitle+'/kmeans_clusters.png')
        plt.close()
        self.plot_score_curve(kmeans_accuracy,"kmeans_accuracy_"+dataTitle)


    def kmeansFitBestModel(self, col1, col2, dataTitle, x, k):
        km = KMeans(n_clusters=k, random_state=123)
        km_val = km.fit(x)
        x['cluster'] = km.predict(x)
        cmap = plt.cm.get_cmap('viridis_r')
        ci = 1
        c = x.columns.tolist()[col1]
        c2 = x.columns.tolist()[col2]
        f, ax = plt.subplots()
        for i, cluster in x.groupby('cluster'):
            _ = ax.scatter(cluster[c], cluster[c2], c=np.array([cmap(i / k)]), label=i)
        ax.legend()
        plt.xlabel(c)
        plt.ylabel(c2)
        plt.savefig('image/' + dataTitle + '/kmeans_' + str(ci) + '.png')
        plt.close()
        ci = ci + 1

    def emCluster(self,dataTitle, x, y, c1, c2):

        bic = []
        k_range = [1, 2, 5, 10, 15]
        lowest_bic = np.infty
        best_em = None
        for k in k_range:
            em = GaussianMixture(n_components=k, random_state=123)
            em.fit(x.iloc[:, :-1])
            bic.append(em.bic(x.iloc[:, :-1]))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_em = em
        plt.figure()
        plt.plot(k_range, bic, 'bx-')
        plt.xlabel('n_components')
        plt.ylabel('BIC')
        plt.savefig('image/'+dataTitle+'/em_cluster_comp.png')
        plt.close()


    def emFitBestModel(self, c1, c2, dataTitle, k, x):
        cmap = plt.cm.get_cmap('viridis_r')
        em_cluster = GaussianMixture(n_components=k, random_state=123).fit_predict(x.iloc[:, :-1])
        x['em_cluster'] = em_cluster
        ci = 1
        c = x.columns.tolist()[c1]
        c2 = x.columns.tolist()[c2]
        f, ax = plt.subplots()
        for i, cluster in x.groupby('em_cluster'):
            _ = ax.scatter(cluster[c], cluster[c2], c=np.array([cmap(i / k)]), label=i)
        ax.legend()
        plt.xlabel(c)
        plt.ylabel(c2)
        plt.savefig('image/' + dataTitle + '/em_cluster_' + str(ci) + '.png')
        ci = ci + 1

    def cluster_accuracy(self, original, cluster_label):

        prediction = np.empty_like(original)
        for cl in set(cluster_label):
            mask = cluster_label == cl
            target = Counter(original[mask]).most_common(1)[0][0]
            prediction[mask] = target
        return accuracy_score(original, prediction)

    def plot_score_curve(self, data, title):

        fig = plt.figure()

        num_clusters = list(data.keys())
        score_clusters = list(data.values())
        ax = fig.add_subplot(111, xlabel='# Clusters', ylabel='Score', title=title)

        ax.plot(num_clusters, score_clusters, 'o-', color="b",
                label="Num of Clusters")
        ax.set_xticks(num_clusters)

        ax.legend(loc="best")
        fig.savefig("image/" + title + ".png")  # save the figure to file
        plt.close(fig)
        return plt


def main():
    cluster = Clustering()
    data_file = "data/winequality-red.csv"
    x, y = extractData(data_file)
    cluster.kMeansCluster('wine',x,y,6,10)
    cluster.kmeansFitBestModel(6, 10, 'wine', x, 5)

    cluster.emCluster('wine',x,y,6,10)
    cluster.emFitBestModel(6, 10, 'wine', 5, x)

    data_file = "data/default_of_credit_card_clients.csv"
    x, y = extractData(data_file)
    x = pd.DataFrame(preprocessing.scale(x),columns=x.columns)
    bill_cols = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
    pmt_cols = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
    num_cols=x.shape[1]
    x['total_bill'] = x.loc[:,bill_cols].sum(axis=1)
    x['total_pmt'] = x.loc[:, pmt_cols].sum(axis=1)

    cluster.kMeansCluster('default', x, y, num_cols, num_cols+1)
    cluster.kmeansFitBestModel(num_cols, num_cols+1, 'default', x,10)

    cluster.emCluster('default', x, y, num_cols, num_cols+1)
    cluster.emFitBestModel(num_cols, num_cols+1, 'default', 5, x)


if __name__ == "__main__":
    main()
    exit()
