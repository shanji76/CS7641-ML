from sklearn.decomposition import PCA, FastICA, truncated_svd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import SVC

from Utility import extractData
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from sys import exit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import scipy.sparse as sps
from scipy.linalg import pinv

class Decomposing:

    def pca_dim_reduction(self, x,y, title, dim_max):
        print('PCA')
        exp_vars = []
        for n in range(1,dim_max):
            pca = PCA(n_components=n, random_state=123)
            pca_result = pca.fit_transform(x,y)
            exp_vars.append(sum(pca.explained_variance_ratio_))

        plt.figure()
        plt.plot(range(1,dim_max), exp_vars, 'bx-')
        plt.xlabel('n_components')
        plt.ylabel('Exp Variance')
        plt.savefig('image/' + title + '/pca_train.png')
        plt.close()

    def pca_eval(self, x,y,title,n,class_title):
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
        ax.legend(title=class_title)
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

    def ica_eval(self, x,y,title,n,class_title):
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
        ax.legend(title=class_title)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.savefig('image/' + title + '/ica.png')
        plt.close()

    def rp_dim_reduction(self, x,y, title, dim_max):
        print('Random Projections')
        rp_error = []
        scaler = StandardScaler()
        scaler.fit(x)
        n_min_reconstruction_error = -1
        components_min_error = None
        min_error = float("inf")
        for n in range(2,dim_max+1):
            rp = SparseRandomProjection(n_components=n, random_state=123)
            rp_result = rp.fit_transform(x)
            reconstruction_error = self.reconstructionError(rp,x)
            rp_error.append(reconstruction_error)

            if reconstruction_error < min_error:
                n_min_reconstruction_error = n
                min_error = reconstruction_error


        plt.figure()
        plt.plot(range(1,dim_max), rp_error, 'bx-')
        plt.xlabel('n_components')
        plt.ylabel('Reconstructed Error')
        plt.savefig('image/' + title + '/rp_train.png')
        plt.close()

        return n_min_reconstruction_error

    def rp_eval(self, x,y,title,n,class_title):
        rp = SparseRandomProjection(n_components=n, random_state=123)
        rp_result = rp.fit_transform(x)
        cols = []
        for c in range(1,n+1):
            cols.append('PC'+str(c))

        pc_df = pd.DataFrame(data=rp_result,
                             columns=cols)
        pc_df.head()
        classes = np.sort(np.unique(y))
        f, ax = plt.subplots()
        labels = np.unique(y)
        for i, c, label in zip(classes, 'rgbcmykw', labels):
            plt.scatter(rp_result[y == i, 0], rp_result[y == i, 1],
                        c=c, label=label)
        ax.legend(title=class_title)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.savefig('image/' + title + '/rp.png')
        plt.close()


    def sk_dim_reduction(self, x,y, title, dim_max):
       print('Select K Best')
       pipeline = Pipeline([('scl', StandardScaler()),
                            ('sel', SelectKBest()),
                            ('clf', SVC(kernel='linear', random_state=1))])

       param_grid = [{'sel__k': range(2,dim_max,2),
                      'clf__C': [0.1],
                      'clf__kernel': ['linear']}]
       grid_search = GridSearchCV(pipeline,
                                  param_grid=param_grid,
                                  verbose=1,
                                  scoring='accuracy',
                                  n_jobs=5)
       grid_search.fit(x, y)
       optimal_k = grid_search.best_estimator_.named_steps['sel'].k
       print('Select K Best - optimal k : {}'.format(str(optimal_k)))

       sk = SelectKBest(k=optimal_k)
       sk.fit(x,y)
       # Get the raw p-values for each feature, and transform from p-values into scores
       scores = -np.log10(sk.pvalues_)
       features = x.columns[sk.get_support()]
       # Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
       plt.figure()
       plt.bar(range(len(x.columns)), scores)
       plt.xticks(range(len(x.columns)), range(len(x.columns)), rotation=60)
       plt.xlabel('Features')
       plt.ylabel('Score')
       plt.savefig('image/' + title + '/sk.png')

    def reconstructionError(self, projections, X):
        W = projections.components_
        if sps.issparse(W):
            W = W.todense()
        p = pinv(W)
        reconstructed = ((p @ W) @ (X.T)).T  # Unproject projected data
        errors = np.square(X.values - reconstructed)
        return np.nanmean(errors)

def main():
    data_file = "data/winequality-red.csv"
    x, y = extractData(data_file)
    decompose = Decomposing()
    decompose.pca_dim_reduction(x,y,'wine',11)
    decompose.pca_eval(x,y,'wine',5,'Rating')

    decompose.ica_dim_reduction(x,y,'wine',11)
    decompose.ica_eval(x, y, 'wine', 11,'Rating')

    n = decompose.rp_dim_reduction(x, y, 'wine', 11)
    print('Optimal n for RP(Wine data set) : {} '.format(str(n)))
    decompose.rp_eval(x, y, 'wine', n,'Rating')

    k = decompose.sk_dim_reduction(x, y, 'wine', 11)
    # decompose.sk_eval(x, y, 'wine', k)

    data_file = "data/default_of_credit_card_clients.csv"
    x, y = extractData(data_file)
    decompose = Decomposing()
    decompose.pca_dim_reduction(x, y, 'default', 24)
    decompose.pca_eval(x, y, 'default', 10,'Defaulted')

    decompose.ica_dim_reduction(x, y, 'default', 24)
    decompose.ica_eval(x, y, 'default', 6, 'Defaulted')

    n = decompose.rp_dim_reduction(x, y, 'default', 24)
    print('Optimal n for RP(CC Default data set) : {} '.format(str(n)))
    decompose.rp_eval(x, y, 'default', n, 'Defaulted')

    k = decompose.sk_dim_reduction(x, y, 'default', 24)

if __name__ == "__main__":
    main()
    exit()
