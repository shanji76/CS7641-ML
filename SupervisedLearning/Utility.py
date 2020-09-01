import pandas as pd


def extractData(data_file):
       data = pd.DataFrame()
       chunksize = 10 ** 5
       for chunk in pd.read_csv(data_file, chunksize=chunksize):
           data = data.append(chunk)
           break
       X = data.iloc[:, :-1]
       Y = data.iloc[:, -1]
       return X, Y