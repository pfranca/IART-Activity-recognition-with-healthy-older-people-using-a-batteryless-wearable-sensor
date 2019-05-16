#/usr/bin/python3

import tensorflow as tf
import pandas as pd
import os


csv_path = "/home/franza/Desktop/WIP/IART-Activity-recognition-with-healthy-older-people-using-a-batteryless-wearable-sensor/dataset/d1p01M"
#train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(csv_path), origin = csv_path)

train_dataset_fp = pd.read_csv(csv_path)


column_names = ['time', 'acc frontal', 'acc vertical', 'acc lateral', 'antenaId', 'RSSI', 'Phase', 'Freq', 'activity']
features_names = column_names[:-1]
label_name = column_names[-1]

batch_size = 401

train_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1
)

features, labels = next(iter(train_dataset))

print(features)