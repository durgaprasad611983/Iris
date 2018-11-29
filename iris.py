#!/opt/conda/bin

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import pandas as pd

Iris = pd.read_csv('/opt/conda/bin/Iris/Iris.csv')

x = Iris.iloc[:,:4]
y = Iris.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(x,y) 
rfc = RandomForestClassifier(n_estimators=100,n_jobs=2)
rfc.fit(X_train,y_train)

Prediction = rfc.predict(X_test)
print(Prediction)
