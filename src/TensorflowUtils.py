import tensorflow as tf

def batch_dataset(dataset, take_count, batch_count, shuffle_count=None):
  """
  Shuffle and batch the data
  :param dataset: the dataset
  :param take_count: how many samples are kept
  :param batch_count: Batch count
  :param shuffle_count: Shuffle count
  :return: dataset batched
  """
  # For perfect shuffling, a buffer size greater 
  # than or equal to the full size of the dataset is required.
  if shuffle_count is None:
    shuffle_count = take_count
  
  batched_dataset = dataset.take(take_count).shuffle(shuffle_count, seed=42).batch(batch_count)
  return batched_dataset

def save(model, path):
  """
  export saved model
  :param model: model to save
  :param path: file path
  :return: void
  """
  signature_dict = {'model': tf.function(model, input_signature = [model.input_signature])}
  tf.saved_model.save(model, path, signature_dict)

def fit(model, train_dataset_batch, test_dataset_batch, optimizer_function, compute_loss, epochs):
  train_losses = []
  train_accurarcies = []

  test_losses = []
  test_accurarcies = []

  for epoch in range(epochs):
    train_loss_aggregate = tf.keras.metrics.Mean(name="train_loss")
    test_loss_aggregate = tf.keras.metrics.Mean(name="test_loss")
    train_accuracy_aggregate = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    test_accuracy_aggregate = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    for train_images, train_labels in train_dataset_batch:
      with tf.GradientTape() as tape:
        # forward propagation
        predictions = model(train_images)
        # calculate loss
        loss = compute_loss(train_labels, predictions)
        
      # calculate gradients from model definition and loss
      gradients = tape.gradient(loss, model.trainable_variables)
      # update model from gradients
      optimizer_function.apply_gradients(zip(gradients, model.trainable_variables))

      train_loss_aggregate(loss)
      train_accuracy_aggregate(train_labels, predictions)

    for test_images, test_labels in test_dataset_batch:
      predictions = model(test_images)

      loss = compute_loss(test_labels, predictions)

      test_loss_aggregate(loss)
      test_accuracy_aggregate(test_labels, predictions)

    train_losses.append(train_loss_aggregate.result().numpy())
    train_accurarcies.append(train_accuracy_aggregate.result().numpy()*100)

    test_losses.append(test_loss_aggregate.result().numpy())
    test_accurarcies.append(test_accuracy_aggregate.result().numpy()*100)

    print('epoch', epoch,
          'train loss', train_losses[-1],
          'train accuracy', train_accurarcies[-1],
          'test loss', test_losses[-1],
          'test accuracy', test_accurarcies[-1])

    save(model, 'mnist/epoch/{0}'.format(epoch))
    
    if epoch > 1:
      if test_accurarcies[-2] >= test_accurarcies[-1] and test_accurarcies[-3] >= test_accurarcies[-2]:
        break
  
  return (train_losses, test_losses), (train_accurarcies, test_accurarcies), epoch