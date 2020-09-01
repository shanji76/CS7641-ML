from DecisionTree import DecisionTree
from NeurlNetwork import NeuralNetwork


def main():
    treeClassify = DecisionTree()
    print('------- DecisionTree - Classification for : WineQuality-Red -------')
    treeClassify.classify("winequality-red.csv", encode=False)
    print('------- DecisionTree with Boost - Classification for : WineQuality-Red -------')
    treeClassify.classifyWithBoost("winequality-red.csv", encode=False)
    print('------- DecisionTree - Classification for : Fraud detection -------')
    treeClassify.classify("fraud_detection_short.csv", encode=True)
    print('------- DecisionTree with Boost - Classification for : Fraud detection -------')
    treeClassify.classifyWithBoost("fraud_detection_short.csv", encode=True)

    neuralNw = NeuralNetwork()
    print('------- NeuralNetwork : Classification for : WineQuality-Red -------')
    neuralNw.classify("winequality-red.csv", encode=False)
    print('------- NeuralNetwork: Classification for : Fraud detection -------')
    neuralNw.classify("fraud_detection_short.csv", encode=True)

if __name__ == "__main__":
    main()