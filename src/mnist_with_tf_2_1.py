# -*- coding: utf-8 -*-
"""
MNIST with tf 2.1
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Reshape, MaxPooling2D
from tensorflow.keras import Model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

from DataDistribution import *
from DataStatistic import *
from MnistUtils import *
from SimpleConvModel import *
from TensorflowUtils import *

print(tf.__version__)

# Load
mnist_dataset = tf.keras.datasets.mnist.load_data()
# Unpack
(x_train, y_train), (x_test, y_test) = mnist_dataset

# Train dataset shapes
print('Train X shape ', x_train.shape)
print('Train Y shape ', y_train.shape)

# Test dataset shapes
print('Test X shape ', x_test.shape)
print('Test Y shape ', y_test.shape)

m_train = x_train.shape[0]
m_test = x_test.shape[0]
print("train examples", m_train,
      "\ntest examples", m_test)

print_statistic(x_train, "pixel")

print_statistic(y_train, "label")

traintest = dataset_distributions(y_train, y_test)
print(traintest)

print_dataset_distributions(traintest)

show_image(x_train, 0)

optimizer_function = tf.keras.optimizers.Adam()
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

def compute_loss(labels, logits):
  """
  Compute loss
  :param labels: true label
  :param logits: predicted label
  :return: loss
  """
  return loss_function(labels, logits)

# Create a model instance
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28

model = SimpleConvModel(IMAGE_HEIGHT, IMAGE_WIDTH, 1)
model.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
model.summary()

# Prepare data
(x_train, y_train), (x_test, y_test) = prepare_mnist_dataset(mnist_dataset)

# Train
# Get a `TensorSliceDataset` object from `ndarray`s
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset_batch = batch_dataset(train_dataset,
                        take_count = 60000,
                        batch_count = 100)

# Test
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset_batch = test_dataset.batch(100)

EPOCHS = 10
(train_losses, test_losses), (train_accurarcies, test_accurarcies), epoch_stop = fit(model, train_dataset_batch, test_dataset_batch, optimizer_function, compute_loss, EPOCHS)

# Train/Test losses
fig, ax = plt.subplots(figsize = (14,10))
plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Loss against epoch')
ax.legend(loc='upper right')

plt.savefig('losses.png')

# Train/Test accuracies
fig, ax = plt.subplots(figsize = (14,10))
plt.plot(train_accurarcies, label='train')
plt.plot(test_accurarcies, label='test')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy against epoch')
ax.legend(loc='upper left')

plt.savefig('accuracies.png')

# Show confusion matrices for epochs
valueMax = 0
for epoch in range(epoch_stop):
  savedModel = tf.saved_model.load("mnist/epoch/{0}".format(epoch))
  confusion = custom_confusion_matrix(savedModel, x_test, y_test)
  if valueMax == 0:
    valueMax = np.max(confusion)
  
  print_confusion(confusion, valueMax, epoch)

# Show convolution for one image
sample_index = 10
show_image(x_test, sample_index)

prediction = model(x_test[sample_index].reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))

print("label", y_test[sample_index])
print("prediction", tf.math.argmax(prediction, axis=1).numpy())

model.print_convolutions()
model.print_max_poolings()

# Error analysis
# Load model
epoch = 5
savedModel = tf.saved_model.load("mnist/epoch/{0}".format(epoch))

# calculate prediction
m = x_test.shape[0]
test_predictions = model(x_test.reshape(m, IMAGE_HEIGHT, IMAGE_WIDTH, 1))

# Build error sets
error_indices = y_test != np.argmax(test_predictions, axis=1)

error_images = x_test[error_indices]
error_labels = y_test[error_indices]
error_predictions = test_predictions[error_indices]

errors = pd.DataFrame({'label': error_labels, 'prediction': np.argmax(error_predictions, axis=1)})
print(errors)

np.set_printoptions(suppress=True)

# Show errors
for i, error_prediction in enumerate(error_predictions):
  print("index", i)
  print("true label", error_labels[i])
  print("prediction", tf.math.argmax(error_prediction).numpy())

  top3 = tf.math.top_k(error_prediction, 3)
  print("top k", top3.indices.numpy(), top3.values.numpy())

  show_image(error_images, i)

# Show convolutions for one prediction error
sample_index = 128
model(error_images[sample_index].reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))

model.print_convolutions()
model.print_max_poolings()
