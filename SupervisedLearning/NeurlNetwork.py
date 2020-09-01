from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from Utility import extractData


class NeuralNetwork:

    def classify(self, data_file, encode):
        X, Y = extractData(data_file)

        enc = LabelEncoder()

        if encode:
            rows, cols = X.shape
            for c in range(cols):
                if not str(X.iloc[1][c]).isnumeric():
                    enc.fit(X.iloc[:, c])
                    X.iloc[:, c] = enc.transform(X.iloc[:, c])
            enc.fit(Y)
            Y = enc.transform(Y)

        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3)

        classifier = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(50,),random_state=123,activation='tanh',max_iter=300)
        classify_model = classifier.fit(train_x, train_y)
        pred_y = classify_model.predict(test_x)
        accuracy = metrics.accuracy_score(test_y, pred_y) * 100

        print('Accuracy of Neural Network = {:.2f}%'.format(accuracy))

def main():
    neuralNw = NeuralNetwork()
    print('------- Classification for : WineQuality-Red -------')
    neuralNw.classify("winequality-red.csv", encode=False)
    print('------- Classification for : Diabetes detection -------')
    neuralNw.classify("diabetes_data_upload.csv", encode=True)

if __name__ == "__main__":
    main()