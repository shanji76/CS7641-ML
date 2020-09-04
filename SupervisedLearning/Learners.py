from DecisionTree import DecisionTree
from KNeighbours import KNNClassifier
from NeurlNetwork import NeuralNetwork
from SupportVectorMachines import SupportVectorMachine


def main():
    treeClassify = DecisionTree()
    print('------- DecisionTree - Classification for : WineQuality-Red -------')
    treeClassify.classify("winequality-red.csv", encode=False)
    print('------- DecisionTree with Boost - Classification for : WineQuality-Red -------')
    treeClassify.classifyWithBoost("winequality-red.csv", encode=False)
    print('------- DecisionTree - Classification for : Fraud detection -------')
    treeClassify.classify("default_of_credit_card_clients.csv", encode=False)
    print('------- DecisionTree with Boost - Classification for : Fraud detection -------')
    treeClassify.classifyWithBoost("default_of_credit_card_clients.csv", encode=False)

    neuralNw = NeuralNetwork()
    print('------- NeuralNetwork : Classification for : WineQuality-Red -------')
    neuralNw.classify("winequality-red.csv", encode=False)
    print('------- NeuralNetwork: Classification for : Fraud detection -------')
    neuralNw.classify("default_of_credit_card_clients.csv", encode=False)

    svcClassify = SupportVectorMachine()
    print('------- SVM - Classification for : WineQuality-Red -------')
    svcClassify.classify("winequality-red.csv", encode=False, kernel='rbf')
    svcClassify.classify("winequality-red.csv", encode=False, kernel='linear')
    print('------- SVM - Classification for : Fraud detection -------')
    svcClassify.classify("default_of_credit_card_clients.csv", encode=False, kernel='rbf')
    svcClassify.classify("default_of_credit_card_clients.csv", encode=False, kernel='linear')

    knnClassify = KNNClassifier()
    print('------- KNN - Classification for : WineQuality-Red -------')
    knnClassify.classify("winequality-red.csv", encode=False, k=6)
    knnClassify.classify("winequality-red.csv", encode=False, k=10)
    print('------- KNN - Classification for : Fraud detection -------')
    knnClassify.classify("default_of_credit_card_clients.csv", encode=False, k=6)
    knnClassify.classify("default_of_credit_card_clients.csv", encode=False, k=7)

if __name__ == "__main__":
    main()