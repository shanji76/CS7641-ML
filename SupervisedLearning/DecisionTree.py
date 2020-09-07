from sklearn import tree, metrics
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from Utility import extractData, plotPerformance



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
        train_x, test_x, train_y,test_y = train_test_split(X,Y, test_size=0.3, random_state=123)


        classifier = tree.DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=3, random_state=22)
        classify_model =  classifier.fit(train_x, train_y)
        cv = cross_validate(classifier, train_x, train_y,)
        pred_y = classify_model.predict(test_x)
        accuracy = metrics.accuracy_score(test_y, pred_y) * 100

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

        classify_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, max_depth = 2, random_state = 12).fit(train_x, train_y)
        pred_y = classify_model.predict(test_x)
        classify_model.cr
        accuracy = classify_model.score(test_x, test_y)*100

        print('Accuracy of GradientBoostingClassifier = {:.2f}%'.format(accuracy))

        plotPerformance(test_y, pred_y, label, 'Algorithm: Decision Tree with Boost')




