from DecisionTree import DecisionTree
from KNeighbours import KNNClassifier
from NeurlNetwork import NeuralNetwork
from SupportVectorMachines import SupportVectorMachine


def main():
    treeClassify = DecisionTree()
    print('------- DecisionTree - Classification for : WineQuality-Red -------')
    treeClassify.classify("winequality-red.csv", encode=False, label='Wine Quality')
    print('------- DecisionTree with Boost - Classification for : WineQuality-Red -------')
    treeClassify.classifyWithBoost("winequality-red.csv", encode=False, label='Wine Quality')
    print('------- DecisionTree - Classification for : CC Default -------')
    treeClassify.classify("default_of_credit_card_clients.csv", encode=False, label='Creditcard Default')
    print('------- DecisionTree with Boost - Classification for : CC Default -------')
    treeClassify.classifyWithBoost("default_of_credit_card_clients.csv", encode=False, label='Creditcard Default')

    neuralNw = NeuralNetwork()
    print('------- NeuralNetwork : Classification for : WineQuality-Red -------')
    neuralNw.classify("winequality-red.csv", encode=False, label='Wine Quality')
    print('------- NeuralNetwork: Classification for : CC Default -------')
    neuralNw.classify("default_of_credit_card_clients.csv", encode=False, label='Creditcard Default')

    svcClassify = SupportVectorMachine()
    print('------- SVM - Classification for : WineQuality-Red -------')
    svcClassify.classify("winequality-red.csv", encode=False, kernel='rbf', label='Wine Quality')
    svcClassify.classify("winequality-red.csv", encode=False, kernel='linear', label='Wine Quality')
    print('------- SVM - Classification for : CC Default -------')
    svcClassify.classify("default_of_credit_card_clients.csv", encode=False, kernel='rbf', label='Creditcard Default')
    svcClassify.classify("default_of_credit_card_clients.csv", encode=False, kernel='linear', label='Creditcard Default')

    knnClassify = KNNClassifier()
    print('------- KNN - Classification for : WineQuality-Red -------')
    knnClassify.classify("winequality-red.csv", encode=False, label='Wine Quality')
    print('------- KNN - Classification for : CC Default -------')
    knnClassify.classify("default_of_credit_card_clients.csv", encode=False, label='Creditcard Default')


if __name__ == "__main__":
    main()