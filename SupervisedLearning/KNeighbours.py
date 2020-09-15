from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Utility import extractData, plotPerformance, getBestModel, plotValidationCurve, plotLearningCurve


class KNNClassifier:

   def classify(self, data_file, encode, label):
        X, Y = extractData(data_file)

        enc = LabelEncoder()

        if encode:
            rows,cols = X.shape
            for c in range(cols):
                if not str(X.iloc[1][c]).isnumeric() :
                    enc.fit(X.iloc[:,c])
                    X.iloc[:,c] = enc.transform(X.iloc[:,c])
            enc.fit(Y)
            Y = enc.transform(Y)

        #
        train_x, test_x, train_y,test_y = train_test_split(X,Y, test_size=0.3, random_state=123)


        knn = KNeighborsClassifier()
        parameter_grid = {'n_neighbors': range(1, 10)}
        classifier, grid_search = getBestModel(knn, parameter_grid, train_x, train_y)
        plotValidationCurve("KNeighbours", label, grid_search, train_x, train_y, parameter_grid)
        plotLearningCurve("KNeighbours", label, classifier, X, Y)
        classify_model =  classifier.fit(train_x, train_y)
        pred_y = classify_model.predict(test_x)
        accuracy = classify_model.score(test_x, test_y) * 100

        print('Accuracy of KNN = {:.2f}%'.format(accuracy))
        plotPerformance(test_y, pred_y, label, 'Algorithm: k-Nearest Neighbors')

def main():
    knnClassify = KNNClassifier()
    print('------- KNN - Classification for : WineQuality-Red -------')
    knnClassify.classify("winequality-red.csv", encode=False, label='Wine Quality')
    print('------- KNN - Classification for : CC Default -------')
    knnClassify.classify("default_of_credit_card_clients.csv", encode=False, label='Creditcard Default')


if __name__ == "__main__":
    main()