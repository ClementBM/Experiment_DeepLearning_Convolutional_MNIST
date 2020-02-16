import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Reshape, MaxPooling2D
import tensorflow as tf

class SimpleConvModel(Model):
  """
  Custom convolutional model
  """
  def __init__(self, image_height, image_width, channel_count):
    """
    Constructor
    :param self: model
    :param image_height: height of image in pixel
    :param image_width: width of image in pixel
    :channel_count: channel count, for example color image has 3 channels, grayscale image has only one channel
    :return: void
    """
    super(SimpleConvModel, self).__init__()

    # Define sequential layers
    self.convolution = Conv2D(32, (3,3), input_shape=(image_height, image_width, channel_count), activation="relu")
    self.max_pooling = MaxPooling2D(2, 2)
    self.flatten = Flatten()
    self.dense = Dense(128, activation="relu")
    self.softmax = Dense(10, activation="softmax")

    # Keep convolutional layer output
    self.convolutional_output = tf.constant(0)
    # Keep max pooling layer output
    self.max_pooling_output = tf.constant(0)
    # Input signature for tf.saved_model.save()
    self.input_signature = tf.TensorSpec(shape=[None, image_height, image_width, channel_count], dtype=tf.float32, name='prev_img')

  def call(self, inputs):
    """
    Forward propagation
    :param self: model
    :param inputs: tensor of dimension [batch_size, image_height, image_width, channel_count]
    :return: predictions
    """
    self.convolutional_output = self.convolution(inputs)
    self.max_pooling_output = self.max_pooling(self.convolutional_output)
    x = self.flatten(self.max_pooling_output)
    x = self.dense(x)
    return self.softmax(x)

  def print_convolutions(self):
    """
    Print convolution outputs
    :param self: model
    :return: void
    """
    print("convolution shape:", self.convolutional_output.shape)
    
    count = self.convolutional_output.shape[-1]

    # print all convolution outputs
    for index in range(0, count):
      convolution = self.convolutional_output[0, :, :, index]
      
      plt.title("Convolution n°{}".format(index + 1))
      plt.imshow(convolution, cmap= plt.get_cmap('gray_r'))
      plt.show()
  
  def print_max_poolings(self):
    """
    Print max pooling outputs
    :param self: model
    :return: void
    """
    print("max pooling shape:", self.max_pooling_output.shape)

    count = self.max_pooling_output.shape[-1]

    # print all max pooling outputs
    for index in range(0, count):
      max_pooling = self.max_pooling_output[0, :, :, index]
      
      plt.title("Max pooling n°{}".format(index + 1))
      plt.imshow(max_pooling, cmap=plt.get_cmap('gray_r'))
      plt.show()