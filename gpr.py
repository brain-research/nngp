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

"""Gaussian process regression model based on GPflow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean("print_kernel", False, "Option to print out kernel")


class GaussianProcessRegression(object):
  """Gaussian process regression model based on GPflow.

  Args:
    input_x: numpy array, [data_size, input_dim]
    output_x: numpy array, [data_size, output_dim]
    kern: NNGPKernel class
  """

  def __init__(self, input_x, output_y, kern):
    with tf.name_scope("init"):
      self.input_x = input_x
      self.output_y = output_y
      self.num_train, self.input_dim = input_x.shape
      _, self.output_dim = output_y.shape

      self.kern = kern
      self.stability_eps = tf.identity(tf.placeholder(tf.float64))
      self.current_stability_eps = 1e-10

      self.y_pl = tf.placeholder(
          tf.float64, [self.num_train, self.output_dim], name="y_train")
      self.x_pl = tf.identity(
          tf.placeholder(tf.float64, [self.num_train, self.input_dim],
                         name="x_train"))

      self.l_np = None
      self.v_np = None
      self.k_np = None

    self.k_data_data = tf.identity(self.kern.k_full(self.x_pl))

  def _build_predict(self, n_test, full_cov=False):
    with tf.name_scope("build_predict"):
      self.x_test_pl = tf.identity(
          tf.placeholder(tf.float64, [n_test, self.input_dim], name="x_test_pl")
      )

    tf.logging.info("Using pre-computed Kernel")
    self.k_data_test = self.kern.k_full(self.x_pl, self.x_test_pl)

    with tf.name_scope("build_predict"):
      a = tf.matrix_triangular_solve(self.l, self.k_data_test)
      fmean = tf.matmul(a, self.v, transpose_a=True)

      if full_cov:
        fvar = self.kern.k_full(self.x_test_pl) - tf.matmul(
            a, a, transpose_a=True)
        shape = [1, 1, self.y_pl.shape[1]]
        fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
      else:
        fvar = self.kern.k_diag(self.x_test_pl) - tf.reduce_sum(tf.square(a), 0)
        fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, self.output_y.shape[1]])

      self.fmean = fmean
      self.fvar = fvar

  def _build_cholesky(self):
    tf.logging.info("Computing Kernel")
    self.k_data_data_reg = self.k_data_data + tf.eye(
        self.input_x.shape[0], dtype=tf.float64) * self.stability_eps
    if FLAGS.print_kernel:
      self.k_data_data_reg = tf.Print(
          self.k_data_data_reg, [self.k_data_data_reg],
          message="K_DD = ", summarize=100)
    self.l = tf.cholesky(self.k_data_data_reg)
    self.v = tf.matrix_triangular_solve(self.l, self.y_pl)

  def predict(self, test_x, sess, get_var=False):
    """Compute mean and varaince prediction for test inputs.

    Raises:
      ArithmeticError: Cholesky fails even after increasing to large values of
        stability epsilon.
    """
    if self.l_np is None:
      self._build_cholesky()
      start_time = time.time()
      self.k_np = sess.run(self.k_data_data,
                           feed_dict={self.x_pl: self.input_x})
      tf.logging.info("Computed K_DD in %.3f secs" % (time.time() - start_time))

      while self.current_stability_eps < 1:
        try:
          start_time = time.time()
          self.l_np, self.v_np = sess.run(
              [self.l, self.v],
              feed_dict={self.y_pl: self.output_y,
                         self.k_data_data: self.k_np,
                         self.stability_eps: self.current_stability_eps})
          tf.logging.info(
              "Computed L_DD in %.3f secs"% (time.time() - start_time))
          break

        except tf.errors.InvalidArgumentError:
          self.current_stability_eps *= 10
          tf.logging.info("Cholesky decomposition failed, trying larger epsilon"
                          ": {}".format(self.current_stability_eps))

    if self.current_stability_eps > 0.2:
      raise ArithmeticError("Could not compute Cholesky decomposition.")

    n_test = test_x.shape[0]
    self._build_predict(n_test)
    feed_dict = {
        self.x_pl: self.input_x,
        self.x_test_pl: test_x,
        self.l: self.l_np,
        self.v: self.v_np
    }

    start_time = time.time()
    if get_var:
      mean_pred, var_pred = sess.run(
          [self.fmean, self.fvar], feed_dict=feed_dict)
      tf.logging.info("Did regression in %.3f secs"% (time.time() - start_time))
      return mean_pred, var_pred, self.current_stability_eps

    else:
      mean_pred = sess.run(self.fmean, feed_dict=feed_dict)
      tf.logging.info("Did regression in %.3f secs"% (time.time() - start_time))
      return mean_pred, self.current_stability_eps

