import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  

csv_path = "/home/franza/Desktop/WIP/IART-Activity-recognition-with-healthy-older-people-using-a-batteryless-wearable-sensor/dataset/teste.csv"

names = ['time', 'frontalAcc', 'verticalAcc', 'laterallAcc', 'CantennaId','rssi', 'phase','freq','label']

# Read dataset to pandas dataframe
dataset = pd.read_csv(csv_path, names=names)  

#dataset.head()  

X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 8].values  

#print("dfdf")

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False, stratify = None)  

print(y_train)

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  