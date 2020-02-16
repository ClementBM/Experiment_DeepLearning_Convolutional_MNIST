import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

def print_statistic(dataset, name):
  """
  Print basic statistics about on dataset
  Including
    - number of unique values
    - min and max values
    - standard deviation (std)
    - mean
  :param dataset: data set
  :return: void
  """
  value_count = len(np.unique(dataset))
  min_value = np.min(dataset)
  max_value = np.max(dataset)
  std = np.std(dataset)
  mean = np.mean(dataset)

  print("{} unique values:".format(name), value_count,
        "\nmin value:", min_value,
        "\nmax value:", max_value,
        "\nstd:", std,
        "\nmean:", mean)

def compute_accuracy(labels, logits):
  """
  Compute accuracy
  :param labels: true label
  :param logits: predicted label
  :return: accuracy of type float
  """
  predictions = tf.math.argmax(logits, axis=1)
  return tf.math.reduce_mean(tf.cast(tf.math.equal(predictions, labels), tf.float32))

def custom_confusion_matrix(model, images, labels):
  """
  Get a custom confusion matrix containing:
    - accuracies on diagonal
    - recalls on the last column
    - precisions on the last row
    - normalized confusion matrix with an arbitrary "m" factor (rounded)
  :param model:
  :param images:
  :param labels:
  :return:
  """
  m = images.shape[0]
  height = images.shape[1]
  width = images.shape[2]

  # calculate prediction
  test_predictions = model(images.reshape(m, height, width, 1))
  confusion = confusion_matrix(labels, np.argmax(test_predictions,axis=1))
  confusion = confusion.astype('float64')

  recalls = np.diagonal(confusion) / np.sum(confusion, axis=0)
  precisions = np.diagonal(confusion) / np.sum(confusion, axis=1)
  accuracy = compute_accuracy(labels, test_predictions)
  precisions = np.append(precisions, accuracy.numpy())

  accuracies = np.diagonal(confusion) / np.sum(confusion, axis=1)

  confusion = np.round(confusion * m / confusion.sum(axis=1)[:, np.newaxis])
  np.fill_diagonal(confusion, accuracies)

  confusion = np.vstack((confusion, recalls))
  confusion = np.column_stack((confusion, precisions))

  return confusion

def print_confusion(confusion, value_max, epoch):
  """
  Print confusion matrix
  :param confusion: confusion matrix
  :param value_max: maximum value for heatmap
  :return: void
  """
  fig = plt.figure(figsize = (14,10))
  heatmap = sns.heatmap(confusion, annot=True, cbar=False, fmt='g', vmin=0, vmax=value_max)
  heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
  heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  
  plt.savefig('confusion-epoch-{}.png'.format(epoch))
