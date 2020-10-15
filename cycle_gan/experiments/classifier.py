import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from cycle_gan.models.cnn import get_encoder

image =  plt.imread("data/train/knifey/knifey-01-0001.jpg")
image.shape
batch_size = 10
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "data/train",
  labels="inferred",
  validation_split=0.2,
  seed=123,
  subset="training",
  image_size=image.shape[:2],
  batch_size=batch_size).map(lambda X, y: (X / 255, tf.one_hot(y, 3)))

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "data/train",
  labels="inferred",
  validation_split=0.2,
  seed=123,
  subset="validation",
  image_size=image.shape[:2],
  batch_size=batch_size).map(lambda X, y: (X / 255, tf.one_hot(y, 3)))

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "data/test",
  labels="inferred",
  image_size=image.shape[:2],
  batch_size=batch_size).map(lambda X, y: (X / 255, tf.one_hot(y, 3)))

model = get_encoder(3, final_activation="softmax")
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics="accuracy")
model.fit(train_ds, epochs=10, validation_data=val_ds)
model.evaluate(test_ds)
for X, y in train_ds:
    print(X.shape, y)

