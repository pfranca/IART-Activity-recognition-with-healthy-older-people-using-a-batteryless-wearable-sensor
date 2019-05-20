import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
#matplotlib inline

csv_path = "/home/franza/Desktop/WIP/IART-Activity-recognition-with-healthy-older-people-using-a-batteryless-wearable-sensor/dataset/teste.csv"

names = ['time', 'frontalAcc', 'verticalAcc', 'laterallAcc', 'CantennaId','rssi', 'phase','freq','label']
# Read dataset to pandas dataframe
dataset = pd.read_csv(csv_path, names=names)  
#print(dataset.shape);
#print(dataset.head());

X = dataset.drop('label', axis=1)  
y = dataset['label']  

#print(y);

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.svm import SVC  
svclassifier = SVC()  
svclassifier.fit(X_train, y_train)  

y_pred = svclassifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  