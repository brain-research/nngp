# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Run experiments with NNGP Kernel.

Usage:

python run_experiments.py \
      --num_train=100 \
      --num_eval=1000 \
      --hparams='nonlinearity=relu,depth=10,weight_var=1.79,bias_var=0.83' \
      --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os.path
import time

import numpy as np
import tensorflow as tf

import gpr
import load_dataset
import nngp

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('hparams', '',
                    'Comma separated list of name=value hyperparameter pairs to'
                    'override the default setting.')
flags.DEFINE_string('experiment_dir', '/tmp/nngp',
                    'Directory to put the experiment results.')
flags.DEFINE_string('grid_path', './grid_data',
                    'Directory to put or find the training data.')
flags.DEFINE_integer('num_train', 1000, 'Number of training data.')
flags.DEFINE_integer('num_eval', 1000,
                     'Number of evaluation data. Use 10_000 for full eval')
flags.DEFINE_integer('seed', 1234, 'Random number seed for data shuffling')
flags.DEFINE_boolean('save_kernel', False, 'Save Kernel do disk')
flags.DEFINE_string('dataset', 'mnist',
                    'Which dataset to use ["mnist"]')
flags.DEFINE_boolean('use_fixed_point_norm', False,
                     'Normalize input variance to fixed point variance')

flags.DEFINE_integer('n_gauss', 501,
                     'Number of gaussian integration grid. Choose odd integer.')
flags.DEFINE_integer('n_var', 501,
                     'Number of variance grid points.')
flags.DEFINE_integer('n_corr', 500,
                     'Number of correlation grid points.')
flags.DEFINE_integer('max_var', 100,
                     'Max value for variance grid.')
flags.DEFINE_integer('max_gauss', 10,
                     'Range for gaussian integration.')


def set_default_hparams():
  return tf.contrib.training.HParams(
      nonlinearity='tanh', weight_var=1.3, bias_var=0.2, depth=2)


def do_eval(sess, model, x_data, y_data, save_pred=False):
  """Run evaluation."""

  gp_prediction, stability_eps = model.predict(x_data, sess)

  pred_1 = np.argmax(gp_prediction, axis=1)
  accuracy = np.sum(pred_1 == np.argmax(y_data, axis=1)) / float(len(y_data))
  mse = np.mean(np.mean((gp_prediction - y_data)**2, axis=1))
  pred_norm = np.mean(np.linalg.norm(gp_prediction, axis=1))
  tf.logging.info('Accuracy: %.4f'%accuracy)
  tf.logging.info('MSE: %.8f'%mse)

  if save_pred:
    with tf.gfile.Open(
        os.path.join(FLAGS.experiment_dir, 'gp_prediction_stats.npy'),
        'w') as f:
      np.save(f, gp_prediction)

  return accuracy, mse, pred_norm, stability_eps


def run_nngp_eval(hparams, run_dir):
  """Runs experiments."""

  tf.gfile.MakeDirs(run_dir)
  # Write hparams to experiment directory.
  with tf.gfile.GFile(run_dir + '/hparams', mode='w') as f:
    f.write(hparams.to_proto().SerializeToString())

  tf.logging.info('Starting job.')
  tf.logging.info('Hyperparameters')
  tf.logging.info('---------------------')
  tf.logging.info(hparams)
  tf.logging.info('---------------------')
  tf.logging.info('Loading data')

  # Get the sets of images and labels for training, validation, and
  # # test on dataset.
  if FLAGS.dataset == 'mnist':
    (train_image, train_label, valid_image, valid_label, test_image,
     test_label) = load_dataset.load_mnist(
         num_train=FLAGS.num_train,
         mean_subtraction=True,
         random_roated_labels=False)
  else:
    raise NotImplementedError

  tf.logging.info('Building Model')

  if hparams.nonlinearity == 'tanh':
    nonlin_fn = tf.tanh
  elif hparams.nonlinearity == 'relu':
    nonlin_fn = tf.nn.relu
  else:
    raise NotImplementedError

  with tf.Session() as sess:
    # Construct NNGP kernel
    nngp_kernel = nngp.NNGPKernel(
        depth=hparams.depth,
        weight_var=hparams.weight_var,
        bias_var=hparams.bias_var,
        nonlin_fn=nonlin_fn,
        grid_path=FLAGS.grid_path,
        n_gauss=FLAGS.n_gauss,
        n_var=FLAGS.n_var,
        n_corr=FLAGS.n_corr,
        max_gauss=FLAGS.max_gauss,
        max_var=FLAGS.max_var,
        use_fixed_point_norm=FLAGS.use_fixed_point_norm)

    # Construct Gaussian Process Regression model
    model = gpr.GaussianProcessRegression(
        train_image, train_label, kern=nngp_kernel)

    start_time = time.time()
    tf.logging.info('Training')

    # For large number of training points, we do not evaluate on full set to
    # save on training evaluation time.
    if FLAGS.num_train <= 5000:
      acc_train, mse_train, norm_train, final_eps = do_eval(
          sess, model, train_image[:FLAGS.num_eval],
          train_label[:FLAGS.num_eval])
      tf.logging.info('Evaluation of training set (%d examples) took '
                      '%.3f secs'%(
                          min(FLAGS.num_train, FLAGS.num_eval),
                          time.time() - start_time))
    else:
      acc_train, mse_train, norm_train, final_eps = do_eval(
          sess, model, train_image[:1000], train_label[:1000])
      tf.logging.info('Evaluation of training set (%d examples) took '
                      '%.3f secs'%(1000, time.time() - start_time))

    start_time = time.time()
    tf.logging.info('Validation')
    acc_valid, mse_valid, norm_valid, _ = do_eval(
        sess, model, valid_image[:FLAGS.num_eval],
        valid_label[:FLAGS.num_eval])
    tf.logging.info('Evaluation of valid set (%d examples) took %.3f secs'%(
        FLAGS.num_eval, time.time() - start_time))

    start_time = time.time()
    tf.logging.info('Test')
    acc_test, mse_test, norm_test, _ = do_eval(
        sess,
        model,
        test_image[:FLAGS.num_eval],
        test_label[:FLAGS.num_eval],
        save_pred=False)
    tf.logging.info('Evaluation of test set (%d examples) took %.3f secs'%(
        FLAGS.num_eval, time.time() - start_time))

  metrics = {
      'train_acc': float(acc_train),
      'train_mse': float(mse_train),
      'train_norm': float(norm_train),
      'valid_acc': float(acc_valid),
      'valid_mse': float(mse_valid),
      'valid_norm': float(norm_valid),
      'test_acc': float(acc_test),
      'test_mse': float(mse_test),
      'test_norm': float(norm_test),
      'stability_eps': float(final_eps),
  }

  record_results = [
      FLAGS.num_train, hparams.nonlinearity, hparams.weight_var,
      hparams.bias_var, hparams.depth, acc_train, acc_valid, acc_test,
      mse_train, mse_valid, mse_test, final_eps
  ]
  if nngp_kernel.use_fixed_point_norm:
    metrics['var_fixed_point'] = float(nngp_kernel.var_fixed_point_np[0])
    record_results.append(nngp_kernel.var_fixed_point_np[0])

  # Store data
  result_file = os.path.join(run_dir, 'results.csv')
  with tf.gfile.Open(result_file, 'a') as f:
    filewriter = csv.writer(f)
    filewriter.writerow(record_results)

  return metrics


def main(argv):
  del argv  # Unused
  hparams = set_default_hparams().parse(FLAGS.hparams)
  run_nngp_eval(hparams, FLAGS.experiment_dir)


if __name__ == '__main__':
  tf.app.run(main)

