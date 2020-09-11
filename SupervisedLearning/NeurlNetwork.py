from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from Utility import extractData, plotPerformance, getBestModel, plotValidationCurve, plotLearningCurve


class NeuralNetwork:

    def classify(self, data_file, encode, label):
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

        nn = MLPClassifier()
        parameter_grid = {'activation' : ['identity', 'logistic', 'tanh', 'relu'],
                            'solver' : ['lbfgs', 'sgd', 'adam'],
                            'hidden_layer_sizes': [
                                (5,),(10,),(15,),(20,)
                            ]
                          }
        classifier, grid_search = getBestModel(nn, parameter_grid, train_x, train_y)
        plotValidationCurve("Neural Network", label, grid_search, train_x, train_y, parameter_grid)
        plotLearningCurve("Neural Network", label, classifier, X, Y)
        classify_model = classifier.fit(train_x, train_y)
        pred_y = classify_model.predict(test_x)
        accuracy = metrics.accuracy_score(test_y, pred_y) * 100

        print('Accuracy of Neural Network = {:.2f}%'.format(accuracy))
        plotPerformance(test_y, pred_y, label, 'Algorithm: Neural Network')

def main():
    neuralNw = NeuralNetwork()
    print('------- NeuralNetwork : Classification for : WineQuality-Red -------')
    neuralNw.classify("winequality-red.csv", encode=False, label='Wine Quality')
    print('------- NeuralNetwork: Classification for : CC Default -------')
    neuralNw.classify("default_of_credit_card_clients.csv", encode=False, label='Creditcard Default')

if __name__ == "__main__":
    main()

