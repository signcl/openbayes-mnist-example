# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import datetime
import os

import openbayestool

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-o", "--output", default="./model", type=str, help="path for output data")
parser.add_argument("-m", "--modelname", default="model", type=str, help="model save target")
parser.add_argument("-e", "--epochs", default=10, type=int, help="epochs")
parser.add_argument("-l", "--logdir", default="./tf_dir", type=str, help="tensorboard data")
args = parser.parse_args()

log_path = os.path.join(args.logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
checkpoint_path = os.path.join(args.output, 'cp.ckpt')
model_path = args.output
model_name = args.modelname

num_classes = 10
epochs = args.epochs
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist \
                                         .load_data()

x_train = x_train.reshape(-1, img_rows * img_cols) / 255.0

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

class OpenBayesMetricsCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        """Print Training Metrics"""
        if batch % 5000 == 0:
          if tf.__version__.startswith('2'):
            openbayestool.log_metric('acc', float(logs.get('accuracy')))
          else:
            openbayestool.log_metric('acc', float(logs.get('acc')))
          openbayestool.log_metric('loss', float(logs.get('loss')))


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
          callbacks=[cp_callback, tb_callback, OpenBayesMetricsCallback()])
model.save(os.path.join(model_path, model_name))
print('done')
