import pandas as pd
import matplotlib.pyplot as plt

def dataset_distributions(training_set, test_set):
  """
  Get distributions of the training and test set
  :param training_set: training set
  :param test_set: test set
  :return: distributions in percentage in a dataframe
  """
  training_set_df = pd.DataFrame({'label': training_set})
  test_set_df = pd.DataFrame({'label': test_set})
  
  m_train = training_set.shape[0]
  m_test = test_set.shape[0]

  train_distribution = training_set_df.groupby(["label"], as_index=False)["label"].size() * 100 / m_train
  test_distribution = test_set_df.groupby(["label"], as_index=False)["label"].size()* 100 / m_test

  train_test_distributions = pd.DataFrame({'train': train_distribution, 'test': test_distribution})
  return train_test_distributions

def print_dataset_distributions(dataset_distributions):
  """
  Print an histogram from dataset distributions
  :param dataset_distributions: dataframe with distributions
  :return: void
  """
  # x locations
  labels = dataset_distributions.index
  # width of the bars
  width = 0.35

  fig, ax = plt.subplots(figsize = (14,10))
  ax.bar(labels - width / 2, dataset_distributions['train'], width, label='train')
  ax.bar(labels + width / 2, dataset_distributions['test'], width, label='test')

  ax.set_ylabel('%')
  ax.set_xlabel('Label')
  ax.set_xticks(labels)
  ax.set_xticklabels(labels)
  ax.legend(loc='lower right')
  ax.set_title('Distribution of labels across training and test set')

  plt.savefig('label-distributions.png')
