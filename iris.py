import pandas as pd
import numpy as np
import pickle

data = pd.read_csv("data.csv")

data.isnull().sum()

y = data["label"]
x = data.drop(["label"],axis = 1)

from sklearn.model_selection import train_test_split
x_train , x_test ,  y_train , y_test =  train_test_split(x,y, test_size = 0.3 , random_state = 0)




#from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier()
sv = forest.fit(x_train, y_train)

pickle.dump(sv, open('iri.pkl', 'wb'))
