from sklearn import tree
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from Utility import extractData, plotPerformance, getBestModel, plotLearningCurve, plotValidationCurve
import pandas as pd

class DecisionTree:

   def classify(self, data_file, encode, label):
        # data = pd.read_csv(data_file)
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
        train_x, test_x, train_y,test_y = train_test_split(X,Y, test_size=0.2, random_state=123)


        #Find best model
        dct = tree.DecisionTreeClassifier()
        parameter_grid = {'criterion': ['gini', 'entropy'],
                          'splitter': ['best', 'random'],
                          'max_depth': range(1,10),
                          'max_features': range(1,5),
                          'min_samples_leaf': range(1,5)}
        classifier, grid_search = getBestModel(dct, parameter_grid, train_x, train_y)
        plotValidationCurve("Decision Tree", label, grid_search, train_x, train_y, parameter_grid)
        plotLearningCurve("Decision Tree", label, classifier, X, Y)

        classify_model =  classifier.fit(train_x, train_y)
        pred_y = classify_model.predict(test_x)
        accuracy = grid_search.best_score_ * 100

        print('Accuracy of Decision Tree(depth={}) = {:.2f}%'.format(classify_model.get_depth(),accuracy))

        plotPerformance(test_y, pred_y, label, 'Algorithm: Decision Tree')



   def classifyWithBoost(self, data_file, encode, label):
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

        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=123)
        dctb = GradientBoostingClassifier(random_state=123)
        parameter_grid = {'learning_rate': [0.2, 0.3, 0.5],
                          'max_depth': [2, 3, 4, 5]
                          }
        classifier, grid_search = getBestModel(dctb, parameter_grid, train_x, train_y)
        plotValidationCurve("Decision Tree with Boost", label, grid_search, train_x, train_y, parameter_grid)
        plotLearningCurve("Decision Tree with Boost", label, classifier, X, Y)
        classify_model = classifier.fit(X, Y)
        pred_y = classify_model.predict(test_x)
        accuracy = classify_model.score(test_x, test_y)*100

        print('Accuracy of GradientBoostingClassifier = {:.2f}%'.format(accuracy))

        plotPerformance(test_y, pred_y, label, 'Algorithm: Decision Tree with Boost')

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

if __name__ == "__main__":
    main()

