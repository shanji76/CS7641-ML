from sklearn.cluster import KMeans
from Utility import extractData
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
        plt.show()

        km = KMeans(n_clusters=4,random_state=123)
        km_val = km.fit(x)
        x['cluster'] = km.predict(x)
        plt.scatter(x.iloc[:,0],y,c=km.labels_, cmap="jet")
        plt.legend()
        plt.show()


def main():
    kmeans = KMeanClustering()
    kmeans.cluster("data/winequality-red.csv")


if __name__ == "__main__":
    main()
