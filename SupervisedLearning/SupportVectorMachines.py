from sklearn.svm import SVC, LinearSVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Utility import extractData, plotPerformance


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


        classifier = self.getSVCForKernel(kernel)
        classify_model =  classifier.fit(train_x, train_y)
        pred_y = classify_model.predict(test_x)
        accuracy = classify_model.score(test_x, test_y) * 100

        print('Accuracy of SVC(kernel={}) = {:.2f}%'.format(kernel,accuracy))

        plotPerformance(test_y, pred_y, label, 'Algorithm: Support Vector Machines(kernel={})'.format(kernel))

   def getSVCForKernel(self, kernel):
       if kernel == 'linear':
           return LinearSVC(random_state=0, tol=1e-5)
       return SVC(kernel=kernel)
