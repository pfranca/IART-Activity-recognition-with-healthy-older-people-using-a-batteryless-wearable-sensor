from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

np.random.seed(3);
np.set_printoptions(threshold=np.inf)
#number of activities
classifications = 4

csv_path = "/home/franza/Desktop/WIP/IART-Activity-recognition-with-healthy-older-people-using-a-batteryless-wearable-sensor/dataset/teste.csv"


#load dataset
dataset = np.loadtxt(csv_path, delimiter=',');

X = dataset[:, :-1];
Y = dataset[:, -1];

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

y_train = keras.utils.to_categorical(y_train-1, classifications);
y_test = keras.utils.to_categorical(y_test-1, classifications);

model = Sequential();
model.add(Dense(8, input_dim=8, activation='sigmoid'));
# model.add(Dense(8, activation='relu'));
# model.add(Dense(6, activation='relu'));
# model.add(Dense(6, activation='relu'));
# model.add(Dense(4, activation='relu'));
# model.add(Dense(2, activation='relu'));
model.add(Dense(classifications, activation='softmax'));

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']);
model.fit(x_train, y_train,  epochs=10, validation_data=(x_test,y_test));


#print(y_test)

