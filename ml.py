import tensorflow as tf
import numpy as np
from colorama import init, Fore, Back, Style
init()

print(tf.__version__)

from tensorflow.contrib.learn.python.learn.datasets import base

# Data files
IRIS_TRAINING = 'iris_training.csv'
IRIS_TEST = 'iris_test.csv'

# Load datasets.
training_set = base.load_csv_with_header(
  filename = IRIS_TRAINING,
  features_dtype = np.float32,
  target_dtype = np.int,
)

test_set = base.load_csv_with_header(
  filename = IRIS_TEST,
  features_dtype = np.float32,
  target_dtype = np.int,
)

# print(Back.GREEN + 'training set data: {0}'.format(training_set.data))
# print(Back.YELLOW + 'training set target: {0}'.format(training_set.target))

# Specify that all features have real-value data
feature_name = 'flower_features'
feature_columns = [tf.feature_column.numeric_column(
  key = feature_name,
  shape = [4],
)]

classifier = tf.estimator.LinearClassifier(
  feature_columns = feature_columns,
  n_classes = 3,
  model_dir = "/tmp/iris_model",
)

def input_fn(dataset):
  def _fn():
    features = { 
      feature_name: tf.constant(dataset.data) 
    }
    label = tf.constant(dataset.target)
    return features, label
  return _fn

# Fit model.
