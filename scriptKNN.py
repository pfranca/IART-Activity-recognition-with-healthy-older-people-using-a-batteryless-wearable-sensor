#!/usr/bin/python
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
import time

print("K-Nearest Neighbors Algorithm")

#csv_path = "/home/franza/Desktop/WIP/IART-Activity-recognition-with-healthy-older-people-using-a-batteryless-wearable-sensor/dataset/teste.csv"

csvTrain = "/home/franza/Desktop/WIP/IART-Activity-recognition-with-healthy-older-people-using-a-batteryless-wearable-sensor/datasetFiles/train.csv"

csvTest = "/home/franza/Desktop/WIP/IART-Activity-recognition-with-healthy-older-people-using-a-batteryless-wearable-sensor/datasetFiles/test.csv"

names = ['time', 'frontalAcc', 'verticalAcc', 'laterallAcc', 'CantennaId','rssi', 'phase','freq','label']

# Read dataset to pandas dataframe
#dataset = pd.read_csv(csv_path, names=names)  

datasetTrain = pd.read_csv(csvTrain, names=names)  
datasetTest = pd.read_csv(csvTest, names=names) 

#dataset.head()  

#X = dataset.iloc[:, :-1].values  
#y = dataset.iloc[:, 8].values  

X_train = datasetTrain.iloc[:, :-1].values
X_test = datasetTest.iloc[:, :-1].values
y_train = datasetTrain.iloc[:, 8].values
y_test = datasetTest.iloc[:, 8].values

#print("dfdf")

#from sklearn.model_selection import train_test_split  
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False, stratify = None)  





start = time.time()

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)
scaler.fit(X_test)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=2)  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix  
print()
print()
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))  

print()
print("Classification Report")
print(classification_report(y_test, y_pred))  

end = time.time()

duration = end - start
print('It took ', duration,'seconds to run.')