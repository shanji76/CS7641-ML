from sklearn import tree, metrics
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("heart.csv")
X= data.iloc[:, :-1]
Y = data.iloc[:, -1]
#
train_x, test_x, train_y,test_y = train_test_split(X,Y, test_size=0.3)

classifier = tree.DecisionTreeClassifier(criterion='gini', max_depth=3)
classify_model =  classifier.fit(train_x, train_y)
pred_y = classify_model.predict(test_x)
accuracy = metrics.accuracy_score(test_y, pred_y) * 100

print('Accuracy of Decision Tree(depth={}) = {:.2f}%'.format(classify_model.get_depth(),accuracy))





