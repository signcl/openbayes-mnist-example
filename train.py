# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-i", "--input", required=True, type=str, help="path for input data")
parser.add_argument("-o", "--output", required=True, type=str, help="path for output data")
parser.add_argument("-m", "--modelname", default="model.h5", type=str, help="model name")
parser.add_argument("-e", "--epochs", required=True, type=int, help="epochs")
args = parser.parse_args()

log_path = args.output
checkpoint_path = os.path.join(args.output, 'cp.ckpt')
model_path = args.output
model_name = args.modelname

num_classes = 10
epochs = args.epochs
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist \
                                         .load_data(os.path.join(args.input, "mnist.npz"))

x_train = x_train.reshape(-1, img_rows * img_cols) / 255.0

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

def create_model():
  model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(img_rows * img_cols,)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
])

  model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

  return model

model = create_model()
model.summary()

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)

model.fit(x_train, y_train,
          epochs=epochs,
          verbose=1,
          callbacks=[cp_callback, tb_callback])
model.save(os.path.join(model_path, model_name))
print('done')
