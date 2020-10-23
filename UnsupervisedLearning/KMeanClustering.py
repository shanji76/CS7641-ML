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
        kmeans_accuracy = {}
        k_range = [1,2,5,10,15]
        for k in k_range :
            km = KMeans(n_clusters=k,
                             n_init=10,
                             random_state=123)
            km.fit(x)
            inertia.append(km.inertia_)
            kmeans_accuracy[k]=self.cluster_accuracy(y,km.predict(x))

        plt.plot(k_range, inertia, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.savefig('image/kmeans_clusters.png')

        self.plot_score_curve(kmeans_accuracy,"kmeans_accuracy")

        k = 5
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
            plt.close()
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
        plt.close()

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
    kmeans = KMeanClustering()
    kmeans.cluster("data/winequality-red.csv")


if __name__ == "__main__":
    main()
