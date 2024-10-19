#!/ usr/bin/env python3

import random
import csv
import numpy as np

def randomcsv():

###100 random integars between 0 and 100
    datatemp = []
    for i in range(0,100):
        n=random.randint(0,100)
        datatemp.append(n)
    temp_array = np.array(datatemp)
    print(temp_array)

###100 random integars between 1 and 100
    datahumid = []
    for i in range(0,100):
        n=random.randint(1,100)
        datahumid.append(n)
    humid_array = np.array(datahumid)
    print(humid_array)

###saving the arrays in a csv file in columns
    np.savetxt('random.csv', [p for p in zip(temp_array, humid_array)],delimiter=',',)
###adding the column names as new row
    new_row = ['temperature','humidity']
    with open('random.csv', 'r', newline='') as csvfile:
        reader=csv.reader(csvfile)
        data=list(reader)
    data.insert(0, new_row)
    with open('random.csv', 'w', newline='') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerows(data)

##calling the function
randomcsv()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=pd.read_csv("random.csv")

def trainmodel():
    X = data.drop("temperature", axis=1)
    y = data['humidity']


    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2)

    model = LogisticRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print( "Accuracy:", accuracy)

trainmodel()
   
