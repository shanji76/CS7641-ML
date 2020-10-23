from sklearn.cluster import KMeans
from Utility import extractData
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from collections import Counter

class KMeanClustering:

    def cluster(self, data_file):
        x,y = extractData(data_file)
        inertia = []
        for k in range(1,10) :
            km = KMeans(n_clusters=k,
                             n_init=10,
                             random_state=123)
            km.fit(x)
            inertia.append(km.inertia_)

        plt.plot(range(1,10), inertia, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.savefig('image/kmeans_clusters.png')

        k = 4
        km = KMeans(n_clusters=k,random_state=123)
        km_val = km.fit(x)
        x['cluster'] = km.predict(x)

        cmap = plt.cm.get_cmap('viridis_r')
        ci = 1
        for c in x.columns:
            f, ax = plt.subplots()
            if c == 'cluster' or c =='alcohol':
                break
            for i, cluster in x.groupby('cluster'):
                _ = ax.scatter(cluster[c], cluster['alcohol'], c=np.array([cmap(i / k)]), label=i)
            ax.legend()
            plt.xlabel(c)
            plt.ylabel('alcohol')
            plt.savefig('image/kmeans_'+str(ci)+'.png')
            ci = ci+1

        bic = []
        lowest_bic = np.infty
        best_em = None
        for k in range(1,10) :
            em = GaussianMixture(n_components=k, random_state=123)
            em.fit(x.iloc[:,:-1])
            bic.append(em.bic(x.iloc[:,:-1]))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_em = em

        plt.plot(range(1,10), bic, 'bx-')
        plt.xlabel('n_components')
        plt.ylabel('BIC')
        plt.savefig('image/em_cluster_comp.png')

        em_cluster = best_em.fit_predict(x.iloc[:,:-1])
        x['em_cluster'] = em_cluster

        ci = 1
        for c in x.columns:
            f, ax = plt.subplots()
            if c == 'cluster' or c == 'alcohol' or c =='em_cluster':
                break
            for i, cluster in x.groupby('em_cluster'):
                _ = ax.scatter(cluster[c], cluster['alcohol'], c=np.array([cmap(i / k)]), label=i)
            ax.legend()
            plt.xlabel(c)
            plt.ylabel('alcohol')
            plt.savefig('image/em_cluster_' + str(ci) + '.png')
            ci = ci + 1

    def cluster_accuracy(self, original, cluster_label):

        prediction = np.empty_like(original)
        for cl in set(cluster_label):
            mask = cluster_label == cl
            target = Counter(original[mask]).most_common(1)[0][0]
            prediction[mask] = target
        return accuracy_score(original, cluster_label)
def main():
    kmeans = KMeanClustering()
    kmeans.cluster("data/winequality-red.csv")


if __name__ == "__main__":
    main()
