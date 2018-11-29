
from flask import Flask, request
import pandas as pd
import urllib, json

app = Flask(__name__)

@app.route('/Iris_Prediction')

def preview_outcome():

    Iris = pd.read_csv('/opt/conda/bin/Iris.csv')

    x = Iris.iloc[:,:4]
    y = Iris.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(x,y) 
    rfc = RandomForestClassifier(n_estimators=100,n_jobs=2)
    rfc.fit(X_train,y_train)

    Prediction = rfc.predict(X_test)
    print(Prediction)

    return "Iris Test Data Results Predicted"

if __name__ == '__main__':
    print("****Starting Server.....")
    
# Run Server

app.run(host='0.0.0.0',port=9070,use_reloader=True)