import matplotlib.pyplot as plt
import numpy as np

def show_image(dataset, image_index):
  """
  Print an image
  :param dataset: (Tensor) dataset
  :param image_index: (integer)
  :return: void
  """
  height = dataset.shape[1]
  width = dataset.shape[2]

  image = dataset[image_index]
  image = image.reshape(height, width)
  plt.imshow(image, cmap=plt.get_cmap('gray_r'))

  plt.show()

def prepare_mnist_dataset(mnist_dataset):
  """
  Get the data form MNIST dataset
  http://yann.lecun.com/exdb/mnist/
  :param mnist_dataset: MNIST dataset
  :return: tuple containing (x_train, y_train), (x_test, y_test)
  """
  (x_train, y_train), (x_test, y_test) = mnist_dataset
  # Reduce the samples from integers
  x_train, x_test = x_train / np.float32(255), x_test / np.float32(255)
  y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)
  # Get the number of training and test samples
  m_train = x_train.shape[0]
  m_test = x_test.shape[0]
  # Get image dimensions
  height = x_test.shape[1]
  width = x_test.shape[2]
  # Reshape adding one dimension for the channel
  x_train = x_train.reshape(m_train, height, width, 1)
  x_test = x_test.reshape(m_test, height, width, 1)
  return (x_train, y_train), (x_test, y_test)
