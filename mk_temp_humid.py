#! /usr/bin/env python3

"""
Generate simple temperature and humidity data, and
apply a simple classifier from sklearn to that data.
"""

import random
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    n_samples = 100
    temp_humid_fname = 'temp_humid.csv'
    prep_temp_humid_data(temp_humid_fname, n_samples)
    train_model(temp_humid_fname)

def prep_temp_humid_data(out_fname, n_samples):
    """Prepare a (for now) trivial file with temperatures (deg F)
    and relative humidities (percent).  Write the output to a csv
    file with the given filename.
    """
    # temperatures range 0F to 100F
    datatemp = []
    for i in range(n_samples):
        n=random.randint(0,100)
        datatemp.append(n)
    temp_array = np.array(datatemp)
    print(temp_array)

    # humidities range from 1% to 100%
    datahumid = []
    for i in range(n_samples):
        n=random.randint(1,100)
        datahumid.append(n)
    humid_array = np.array(datahumid)
    print(humid_array)

    with open(out_fname, 'w') as outf:
        outf.write('temperature,humidity\n')
        for i in range(n_samples):
            outf.write(f'{temp_array[i]}, {humid_array[i]}\n')

# ###saving the arrays in a csv file in columns
#     np.savetxt('random.csv', [p for p in zip(temp_array, humid_array)],delimiter=',',)
# ###adding the column names as new row
#     new_row = ['temperature','humidity']
#     with open('random.csv', 'r', newline='') as csvfile:
#         reader=csv.reader(csvfile)
#         data=list(reader)
#     data.insert(0, new_row)
#     with open('random.csv', 'w', newline='') as csvfile:
#         writer=csv.writer(csvfile)
#         writer.writerows(data)

def train_model(data_fname):
    data = pd.read_csv(data_fname)
    X = data.drop('temperature', axis=1)
    y = data['humidity']

    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2)

    model = LogisticRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

if __name__ == '__main__':
    main()
