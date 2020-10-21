from sklearn.cluster import KMeans
from Utility import extractData
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

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


def main():
    kmeans = KMeanClustering()
    kmeans.cluster("data/winequality-red.csv")


if __name__ == "__main__":
    main()
