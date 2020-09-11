from sklearn.svm import SVC, LinearSVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Utility import extractData, plotPerformance, plotValidationCurve, plotLearningCurve, getBestModel


class SupportVectorMachine:

   def classify(self, data_file, encode, kernel, label):
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
        parameter_grid = { 'gamma': [1e-3, 1e-4],
                           'C': [1, 10] }
        svc = SVC(random_state=0, kernel=kernel)
        classifier, grid_search = getBestModel(svc, parameter_grid, train_x, train_y)
        plotValidationCurve("SVC(kernel="+kernel+")", label, grid_search, train_x, train_y, parameter_grid)
        plotLearningCurve("SVC(kernel="+kernel+")", label, classifier, X, Y)
        classify_model =  classifier.fit(train_x, train_y)
        pred_y = classify_model.predict(test_x)
        accuracy = classify_model.score(test_x, test_y) * 100

        print('Accuracy of SVC(kernel={}) = {:.2f}%'.format(kernel,accuracy))

        plotPerformance(test_y, pred_y, label, 'Algorithm: Support Vector Machines(kernel={})'.format(kernel))

def main():
    svcClassify = SupportVectorMachine()
    print('------- SVM - Classification for : WineQuality-Red -------')
    svcClassify.classify("winequality-red.csv", encode=False, kernel='rbf', label='Wine Quality')
    svcClassify.classify("winequality-red.csv", encode=False, kernel='linear', label='Wine Quality')
    print('------- SVM - Classification for : CC Default -------')
    svcClassify.classify("default_of_credit_card_clients.csv", encode=False, kernel='rbf',
                         label='Creditcard Default')
    svcClassify.classify("default_of_credit_card_clients.csv", encode=False, kernel='linear',
                         label='Creditcard Default')


if __name__ == "__main__":
    main()


