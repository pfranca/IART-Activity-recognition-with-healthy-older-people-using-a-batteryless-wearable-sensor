#/usr/bin/python3

import tensorflow as tf
#import pandas as pd
import numpy as np
import os

np.set_printoptions(threshold=np.inf)

#np.random.seed(3);

csv_path = "/home/franza/Desktop/WIP/IART-Activity-recognition-with-healthy-older-people-using-a-batteryless-wearable-sensor/dataset/teste.csv"


train_dataset = np.genfromtxt(csv_path, delimiter=',');


features = train_dataset[:, :-1];
label = train_dataset[:, -1];

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(8, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
            )
model.fit(features, label, epochs=10)







#train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(csv_path), origin = csv_path)

# train_dataset_fp = pd.read_csv(csv_path)


# column_names = ['time', 'acc frontal', 'acc vertical', 'acc lateral', 'antenaId', 'RSSI', 'Phase', 'Freq', 'activity']
# features_names = column_names[:-1]
# label_name = column_names[-1]

# batch_size = 401

# train_dataset = tf.contrib.data.make_csv_dataset(
#     train_dataset_fp,
#     batch_size,
#     column_names=column_names,
#     label_name=label_name,
#     num_epochs=1
# )

# features, labels = next(iter(train_dataset))

# #print(features)

# features = tf.keras.utils.normalize(features)

#train_dataset = tf.keras.utils.normalize(train_dataset, axis=1);

#val_loss, val_acc = model.evaluate(features, label)