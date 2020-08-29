from sklearn import tree, metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

class DecisionTree:

   def classify(self, data_file, encode):
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


        classifier = tree.DecisionTreeClassifier(criterion='gini', max_depth=15)
        classify_model =  classifier.fit(train_x, train_y)
        pred_y = classify_model.predict(test_x)
        accuracy = metrics.accuracy_score(test_y, pred_y) * 100

        print('Accuracy of Decision Tree(depth={}) = {:.2f}%'.format(classify_model.get_depth(),accuracy))

   def classifyWithBoost(self, data_file, encode):
        data = pd.read_csv(data_file)
        X = data.iloc[:, :-1]
        Y = data.iloc[:, -1]

        enc = LabelEncoder()

        if encode:
            rows, cols = X.shape
            for c in range(cols):
                if not str(X.iloc[1][c]).isnumeric():
                    enc.fit(X.iloc[:, c])
                    X.iloc[:, c] = enc.transform(X.iloc[:, c])
            enc.fit(Y)
            Y = enc.transform(Y)

        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=4)

        classify_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, max_depth = 2, random_state = 12).fit(train_x, train_y)
        accuracy = classify_model.score(test_x, test_y)*100

        print('Accuracy of GradientBoostingClassifier = {:.2f}%'.format(accuracy))


def main():
    treeClassify = DecisionTree()
    print('------- DecisionTree - Classification for : WineQuality-Red -------')
    treeClassify.classify("winequality-red.csv", encode=False)
    print('------- DecisionTree with Boost - Classification for : WineQuality-Red -------')
    treeClassify.classifyWithBoost("winequality-red.csv", encode=False)
    print('------- DecisionTree - Classification for : Diabetes detection -------')
    treeClassify.classify("diabetes_data_upload.csv", encode=True)
    print('------- DecisionTree with Boost - Classification for : Diabetes detection -------')
    treeClassify.classifyWithBoost("diabetes_data_upload.csv", encode=True)

if __name__ == "__main__":
    main()




