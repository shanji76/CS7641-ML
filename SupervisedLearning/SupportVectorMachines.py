from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class SupportVectorMachine:

   def classify(self, data_file, encode, kernel):
        data = pd.read_csv(data_file)
        X= data.iloc[:, :-1]
        Y = data.iloc[:, -1]

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


        classifier = SVC(kernel=kernel)
        classify_model =  classifier.fit(train_x, train_y)
        accuracy = classify_model.score(test_x, test_y) * 100

        print('Accuracy of SVC(kernel={}) = {:.2f}%'.format(classify_model.kernel,accuracy))

def main():
    svcClassify = SupportVectorMachine()
    print('------- SVM - Classification for : WineQuality-Red -------')
    svcClassify.classify("winequality-red.csv", encode=False, kernel='rbf')
    svcClassify.classify("winequality-red.csv", encode=False, kernel='linear')
    print('------- SVM - Classification for : Diabetes detection -------')
    svcClassify.classify("diabetes_data_upload.csv", encode=True, kernel='rbf')
    svcClassify.classify("diabetes_data_upload.csv", encode=True, kernel='linear')

if __name__ == "__main__":
    main()