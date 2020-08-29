from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class KNNClassifier:

   def classify(self, data_file, encode, k):
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


        classifier = KNeighborsClassifier(n_neighbors=k)
        classify_model =  classifier.fit(train_x, train_y)
        accuracy = classify_model.score(test_x, test_y) * 100

        print('Accuracy of SVC(k={}) = {:.2f}%'.format(k,accuracy))

def main():
    knnClassify = KNNClassifier()
    print('------- KNN - Classification for : WineQuality-Red -------')
    knnClassify.classify("winequality-red.csv", encode=False, k=6)
    knnClassify.classify("winequality-red.csv", encode=False, k=10)
    print('------- KNN - Classification for : Diabetes detection -------')
    knnClassify.classify("diabetes_data_upload.csv", encode=True, k=6)
    knnClassify.classify("diabetes_data_upload.csv", encode=True, k=7)

if __name__ == "__main__":
    main()