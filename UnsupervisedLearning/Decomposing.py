from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from Utility import extractData
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from sys import exit

class Decomposing:

    def pca_dim_reduction(self, x,y, title, dim_max):
        print('PCA')
        exp_vars = []
        for n in range(1,dim_max):
            pca = PCA(n_components=n, random_state=123)
            pca_result = pca.fit_transform(x,y)
            exp_vars.append(sum(pca.explained_variance_))

        plt.figure()
        plt.plot(range(1,dim_max), exp_vars, 'bx-')
        plt.xlabel('n_components')
        plt.ylabel('Exp Variance')
        plt.savefig('image/' + title + '/pca_train.png')
        plt.close()

    def pca_eval(self, x,y,title,n):
        pca = PCA(n_components=n, random_state=123)
        pca_result = pca.fit_transform(x)
        cols = []
        for c in range(1,n+1):
            cols.append('PC'+str(c))

        pc_df = pd.DataFrame(data=pca_result,
                             columns=cols)
        pc_df.head()
        classes = np.sort(np.unique(y))
        f, ax = plt.subplots()
        labels = np.unique(y)
        for i, c, label in zip(classes, 'rgbcmykw', labels):
            plt.scatter(pca_result[y == i, 0], pca_result[y == i, 1],
                        c=c, label=label)
        ax.legend()
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.savefig('image/' + title + '/pca.png')
        plt.close()


    def ica_dim_reduction(self, x,y, title, dim_max):
        print('ICA')
        avg_kurtosis = []
        for n in range(1,dim_max):
            ica = FastICA(n_components=n, random_state=123,tol=0.1)
            ica_result = ica.fit_transform(x)
            avg_kurtosis.append(np.mean(np.apply_along_axis(kurtosis, 0, ica_result)**2))

        plt.figure()
        plt.plot(range(1,dim_max), avg_kurtosis, 'bx-')
        plt.xlabel('n_components')
        plt.ylabel('Avg Kurtosis')
        plt.savefig('image/' + title + '/ica_train.png')
        plt.close()

    def ica_eval(self, x,y,title,n):
        ica = FastICA(n_components=n, random_state=123, tol=0.1)
        ica_result = ica.fit_transform(x)
        cols = []
        for c in range(1,n+1):
            cols.append('PC'+str(c))

        pc_df = pd.DataFrame(data=ica_result,
                             columns=cols)
        pc_df.head()
        classes = np.sort(np.unique(y))
        f, ax = plt.subplots()
        labels = np.unique(y)
        for i, c, label in zip(classes, 'rgbcmykw', labels):
            plt.scatter(ica_result[y == i, 0], ica_result[y == i, 1],
                        c=c, label=label)
        ax.legend()
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.savefig('image/' + title + '/ica.png')
        plt.close()

    def rp_dim_reduction(self, x,y, title, dim_max):
        print('Random Projections')
        rp_error = []
        for n in range(1,dim_max):
            rp = GaussianRandomProjection(n_components=n, random_state=123)
            rp_result = rp.fit_transform(x)
            reconstructed_data = rp_result.dot(rp.components_) + x.mean(axis=0)
            reconstruction_error = ((x - reconstructed_data) ** 2).mean()
            rp_error.append(reconstruction_error)

        plt.figure()
        plt.plot(range(1,dim_max), reconstruction_error, 'bx-')
        plt.xlabel('n_components')
        plt.ylabel('Reconstructed Error')
        plt.savefig('image/' + title + '/rp_train.png')
        plt.close()




def main():
    data_file = "data/winequality-red.csv"
    x, y = extractData(data_file)
    decompose = Decomposing()
    decompose.pca_dim_reduction(x,y,'wine',11)
    decompose.pca_eval(x,y,'wine',4)

    decompose.ica_dim_reduction(x,y,'wine',11)
    decompose.ica_eval(x, y, 'wine', 11)

    decompose.rp_dim_reduction(x, y, 'wine', 11)
    # decompose.ica_eval(x, y, 'wine', 11)

if __name__ == "__main__":
    main()
    exit()
