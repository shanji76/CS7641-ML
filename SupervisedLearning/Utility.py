import pandas
import pandas as pd
from matplotlib import pyplot as plt


def extractData(data_file):
       data = pd.read_csv(data_file)
       X = data.iloc[:, :-1]
       Y = data.iloc[:, -1].values
       return X, Y


def plotPerformance(actual, predicted, label, title):
     conf_matrix = pd.crosstab(actual, predicted, rownames=['Actual'], colnames=['Predicted'], margins=True)
     print(conf_matrix)
     print(conf_matrix.to_latex())

     y_max = max(actual)
     plt.figure(figsize=(6, 4))
     plt.scatter(actual, predicted)
     plt.plot([0, y_max], [0, y_max], '--k')
     plt.axis('tight')
     plt.xlabel('True {}'.format(label))
     plt.ylabel('Predicted {}'.format(label))
     plt.tight_layout()
     plt.title(title)
     plt.show()