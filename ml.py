import tensorflow as tf
import numpy as np
from colorama import init, Fore, Back, Style
init()

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
    print(Fore.GREEN + 'feature_name = %s' % tf.constant(dataset.data))
    features = { feature_name: tf.constant(dataset.data) }
    print(Fore.GREEN + 'label = %s' % tf.constant(dataset.target))
    label = tf.constant(dataset.target)
    print(Fore.WHITE)
    return features, label
  return _fn
# raw data -> input function -> feature columns -> model

# Fit model.
classifier.train(
  steps = 1000,
  input_fn = input_fn(training_set),
)
print('fit done')

# Evaluate accuracy
accuracy_score = classifier.evaluate(
  input_fn = input_fn(test_set),
  steps = 100,
)

print(accuracy_score)

# Export the modal for serving
feature_spec = {
  'flower_features': tf.FixedLenFeature(
    shape = [4],
    dtype = np.float32,
  )
}
serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

print('initiate export')
classifier.export_savedmodel(
  export_dir_base = './tmp/iris_model/export',
  serving_input_receiver_fn = serving_fn,
)
print('Export successfully')