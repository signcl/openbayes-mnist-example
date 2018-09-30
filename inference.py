# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-i", "--input", required=True, type=str, help="path for input data")
parser.add_argument("-m", "--modelpath", required=True, type=str, help="model file name")
args = parser.parse_args()

num_classes = 10
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist \
                                         .load_data(os.path.join(args.input, "mnist.npz"))

x_train = x_train.reshape(-1, img_rows * img_cols) / 255.0
x_test = x_test.reshape(-1, img_rows * img_cols) / 255.0

print('x_train shape:', x_train.shape)
print(x_test.shape[0], 'test samples')

model = tf.keras.models.load_model(args.modelpath)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
