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

"""Neural Network Gaussian Process (nngp) kernel computation.

Implementaion based on
"Deep Neural Networks as Gaussian Processes" by
Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S. Schoenholz,
Jeffrey Pennington, Jascha Sohl-Dickstein
arXiv:1711.00165 (https://arxiv.org/abs/1711.00165).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os

import numpy as np
import tensorflow as tf

import interp

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean("use_precomputed_grid", True,
                     "Option to save/load pre-computed grid")
flags.DEFINE_integer(
    "fraction_of_int32", 32,
    "allow batches at most of size int32.max / fraction_of_int32")


class NNGPKernel(object):
  """The iterative covariance Kernel for Neural Network Gaussian Process.

  Args:
    depth: int, number of hidden layers in corresponding NN.
    nonlin_fn: tf ops corresponding to point-wise non-linearity in corresponding
      NN. e.g.) tf.nn.relu, tf.nn.sigmoid, lambda x: x * tf.nn.sigmoid(x), ...
    weight_var: initial value for the weight_variances parameter.
    bias_var: initial value for the bias_variance parameter.
    n_gauss: Number of gaussian integration grid. Choose odd integer, so that
      there is a gridpoint at 0.
    n_var: Number of variance grid points.
    n_corr: Number of correlation grid points.
    use_fixed_point_norm: bool, normalize input to variance fixed point.
      Defaults to False, normalizing input to unit norm over input dimension.
  """

  def __init__(self,
               depth=1,
               nonlin_fn=tf.tanh,
               weight_var=1.,
               bias_var=1.,
               n_gauss=101,
               n_var=151,
               n_corr=131,
               max_var=100,
               max_gauss=100,
               use_fixed_point_norm=False,
               grid_path=None,
               sess=None):
    self.depth = depth
    self.weight_var = weight_var
    self.bias_var = bias_var
    self.use_fixed_point_norm = use_fixed_point_norm
    self.sess = sess
    if FLAGS.use_precomputed_grid and (grid_path is None):
      raise ValueError("grid_path must be specified to use precomputed grid.")
    self.grid_path = grid_path

    self.nonlin_fn = nonlin_fn
    (self.var_aa_grid, self.corr_ab_grid, self.qaa_grid,
     self.qab_grid) = self.get_grid(n_gauss, n_var, n_corr, max_var, max_gauss)

    if self.use_fixed_point_norm:
      self.var_fixed_point_np, self.var_fixed_point = self.get_var_fixed_point()

  def get_grid(self, n_gauss, n_var, n_corr, max_var, max_gauss):
    """Get covariance grid by loading or computing a new one.
    """
    # File configuration for precomputed grid
    if FLAGS.use_precomputed_grid:
      grid_path = self.grid_path
      # TODO(jaehlee) np.save have broadcasting error when n_var==n_corr.
      if n_var == n_corr:
        n_var += 1
      grid_file_name = "grid_{0:s}_ng{1:d}_ns{2:d}_nc{3:d}".format(
          self.nonlin_fn.__name__, n_gauss, n_var, n_corr)
      grid_file_name += "_mv{0:d}_mg{1:d}".format(max_var, max_gauss)

    # Load grid file if it exists already
    if (FLAGS.use_precomputed_grid and
        tf.gfile.Exists(os.path.join(grid_path, grid_file_name))):
      with tf.gfile.Open(os.path.join(grid_path, grid_file_name), "rb") as f:
        grid_data_np = np.load(f)
        tf.logging.info("Loaded interpolation grid from %s"%
                        os.path.join(grid_path, grid_file_name))
        grid_data = (tf.convert_to_tensor(grid_data_np[0], dtype=tf.float64),
                     tf.convert_to_tensor(grid_data_np[1], dtype=tf.float64),
                     tf.convert_to_tensor(grid_data_np[2], dtype=tf.float64),
                     tf.convert_to_tensor(grid_data_np[3], dtype=tf.float64))

    else:
      tf.logging.info("Generating interpolation grid...")
      grid_data = _compute_qmap_grid(self.nonlin_fn, n_gauss, n_var, n_corr,
                                     max_var=max_var, max_gauss=max_gauss)
      if FLAGS.use_precomputed_grid:
        with tf.Session() as sess:
          grid_data_np = sess.run(grid_data)
        tf.gfile.MakeDirs(grid_path)
        with tf.gfile.Open(os.path.join(grid_path, grid_file_name), "wb") as f:
          np.save(f, grid_data_np)

        with tf.gfile.Open(os.path.join(grid_path, grid_file_name), "rb") as f:
          grid_data_np = np.load(f)
          tf.logging.info("Loaded interpolation grid from %s"%
                          os.path.join(grid_path, grid_file_name))
          grid_data = (tf.convert_to_tensor(grid_data_np[0], dtype=tf.float64),
                       tf.convert_to_tensor(grid_data_np[1], dtype=tf.float64),
                       tf.convert_to_tensor(grid_data_np[2], dtype=tf.float64),
                       tf.convert_to_tensor(grid_data_np[3], dtype=tf.float64))

    return grid_data

  def get_var_fixed_point(self):
    with tf.name_scope("get_var_fixed_point"):
      # If normalized input length starts at 1.
      current_qaa = self.weight_var * tf.constant(
          [1.], dtype=tf.float64) + self.bias_var

      diff = 1.
      prev_qaa_np = 1.
      it = 0
      while diff > 1e-6 and it < 300:
        samp_qaa = interp.interp_lin(
            self.var_aa_grid, self.qaa_grid, current_qaa)
        samp_qaa = self.weight_var * samp_qaa + self.bias_var
        current_qaa = samp_qaa

        with tf.Session() as sess:
          current_qaa_np = sess.run(current_qaa)
        diff = np.abs(current_qaa_np - prev_qaa_np)
        it += 1
        prev_qaa_np = current_qaa_np
      return current_qaa_np, current_qaa

  def k_diag(self, input_x, return_full=True):
    """Iteratively building the diagonal part (variance) of the NNGP kernel.

    Args:
      input_x: tensor of input of size [num_data, input_dim].
      return_full: boolean for output to be [num_data] sized or a scalar value
        for normalized inputs

    Sets self.layer_qaa_dict of {layer #: qaa at the layer}

    Returns:
      qaa: variance at the output.
    """
    with tf.name_scope("Kdiag"):
      # If normalized input length starts at 1.
      if self.use_fixed_point_norm:
        current_qaa = self.var_fixed_point
      else:
        current_qaa = self.weight_var * tf.convert_to_tensor(
            [1.], dtype=tf.float64) + self.bias_var
      self.layer_qaa_dict = {0: current_qaa}
      for l in xrange(self.depth):
        with tf.name_scope("layer_%d" % l):
          samp_qaa = interp.interp_lin(
              self.var_aa_grid, self.qaa_grid, current_qaa)
          samp_qaa = self.weight_var * samp_qaa + self.bias_var
          self.layer_qaa_dict[l + 1] = samp_qaa
          current_qaa = samp_qaa

      if return_full:
        qaa = tf.tile(current_qaa[:1], ([input_x.shape[0].value]))
      else:
        qaa = current_qaa[0]
      return qaa

  def k_full(self, input1, input2=None):
    """Iteratively building the full NNGP kernel.
    """
    input1 = self._input_layer_normalization(input1)
    if input2 is None:
      input2 = input1
    else:
      input2 = self._input_layer_normalization(input2)

    with tf.name_scope("k_full"):
      cov_init = tf.matmul(
          input1, input2, transpose_b=True) / input1.shape[1].value

      self.k_diag(input1)
      q_aa_init = self.layer_qaa_dict[0]

      q_ab = cov_init
      q_ab = self.weight_var * q_ab + self.bias_var
      corr = q_ab / q_aa_init[0]

      if FLAGS.fraction_of_int32 > 1:
        batch_size, batch_count = self._get_batch_size_and_count(input1, input2)
        with tf.name_scope("q_ab"):
          q_ab_all = []
          for b_x in range(batch_count):
            with tf.name_scope("batch_%d" % b_x):
              corr_flat_batch = corr[
                  batch_size * b_x : batch_size * (b_x + 1), :]
              corr_flat_batch = tf.reshape(corr_flat_batch, [-1])

              for l in xrange(self.depth):
                with tf.name_scope("layer_%d" % l):
                  q_aa = self.layer_qaa_dict[l]
                  q_ab = interp.interp_lin_2d(x=self.var_aa_grid,
                                              y=self.corr_ab_grid,
                                              z=self.qab_grid,
                                              xp=q_aa,
                                              yp=corr_flat_batch)

                  q_ab = self.weight_var * q_ab + self.bias_var
                  corr_flat_batch = q_ab / self.layer_qaa_dict[l + 1][0]

              q_ab_all.append(q_ab)

          q_ab_all = tf.parallel_stack(q_ab_all)
      else:
        with tf.name_scope("q_ab"):
          corr_flat = tf.reshape(corr, [-1])
          for l in xrange(self.depth):
            with tf.name_scope("layer_%d" % l):
              q_aa = self.layer_qaa_dict[l]
              q_ab = interp.interp_lin_2d(x=self.var_aa_grid,
                                          y=self.corr_ab_grid,
                                          z=self.qab_grid,
                                          xp=q_aa,
                                          yp=corr_flat)
              q_ab = self.weight_var * q_ab + self.bias_var
              corr_flat = q_ab / self.layer_qaa_dict[l+1][0]
            q_ab_all = q_ab

    return tf.reshape(q_ab_all, cov_init.shape, "qab")

  def _input_layer_normalization(self, x):
    """Input normalization to unit variance or fixed point variance.
    """
    with tf.name_scope("input_layer_normalization"):
      # Layer norm, fix to unit variance
      eps = 1e-15
      mean, var = tf.nn.moments(x, axes=[1], keep_dims=True)
      x_normalized = (x - mean) / tf.sqrt(var + eps)
      if self.use_fixed_point_norm:
        x_normalized *= tf.sqrt(
            (self.var_fixed_point[0] - self.bias_var) / self.weight_var)
      return x_normalized

  def _get_batch_size_and_count(self, input1, input2):
    """Compute batch size and number to split when input size is large.

    Args:
      input1: tensor, input tensor to covariance matrix
      input2: tensor, second input tensor to covariance matrix

    Returns:
      batch_size: int, size of each batch
      batch_count: int, number of batches
    """
    input1_size = input1.shape[0].value
    input2_size = input2.shape[0].value

    batch_size = min(np.iinfo(np.int32).max //
                     (FLAGS.fraction_of_int32 * input2_size), input1_size)
    while input1_size % batch_size != 0:
      batch_size -= 1

    batch_count = input1_size // batch_size
    return batch_size, batch_count


def _fill_qab_slice(idx, z1, z2, var_aa, corr_ab, nonlin_fn):
  """Helper method used for parallel computation for full qab."""
  log_weights_ab_unnorm = -(z1**2 + z2**2 - 2 * z1 * z2 * corr_ab) / (
      2 * var_aa[idx] * (1 - corr_ab**2))
  log_weights_ab = log_weights_ab_unnorm - tf.reduce_logsumexp(
      log_weights_ab_unnorm, axis=[0, 1], keep_dims=True)
  weights_ab = tf.exp(log_weights_ab)

  qab_slice = tf.reduce_sum(
      nonlin_fn(z1) * nonlin_fn(z2) * weights_ab, axis=[0, 1])
  qab_slice = tf.Print(qab_slice, [idx], "Generating slice: ")
  return qab_slice


def _compute_qmap_grid(nonlin_fn,
                       n_gauss,
                       n_var,
                       n_corr,
                       log_spacing=False,
                       min_var=1e-8,
                       max_var=100.,
                       max_corr=0.99999,
                       max_gauss=10.):
  """Construct graph for covariance grid to use for kernel computation.

  Given variance and correlation (or covariance) of pre-activation, perform
  Gaussian integration to get covariance of post-activation.

  Raises:
    ValueError: if n_gauss is even integer.

  Args:
    nonlin_fn: tf ops corresponding to point-wise non-linearity in
      corresponding NN. e.g.) tf.nn.relu, tf.nn.sigmoid,
      lambda x: x * tf.nn.sigmoid(x), ...
    n_gauss: int, number of Gaussian integration points with equal spacing
      between (-max_gauss, max_gauss). Choose odd integer, so that there is a
      gridpoint at 0.
    n_var: int, number of variance grid points.get_grid
    n_corr: int, number of correlation grid points.
    log_spacing: bool, whether to use log-linear instead of linear variance
      grid.
    min_var: float, smallest variance value to generate grid.
    max_var: float, largest varaince value to generate grid.
    max_corr: float, largest correlation value to generate grid. Should be
      slightly smaller than 1.
    max_gauss: float, range (-max_gauss, max_gauss) for Gaussian integration.

  Returns:
    var_grid_pts: tensor of size [n_var], grid points where variance are
      evaluated at.
    corr_grid_pts: tensor of size [n_corr], grid points where correlation are
      evalutated at.
    qaa: tensor of size [n_var], variance of post-activation at given
      pre-activation variance.
    qab: tensor of size [n_var, n_corr], covariance of post-activation at
      given pre-activation variance and correlation.
  """
  if n_gauss % 2 != 1:
    raise ValueError("n_gauss=%d should be an odd integer" % n_gauss)

  with tf.name_scope("compute_qmap_grid"):
    min_var = tf.convert_to_tensor(min_var, dtype=tf.float64)
    max_var = tf.convert_to_tensor(max_var, dtype=tf.float64)
    max_corr = tf.convert_to_tensor(max_corr, dtype=tf.float64)
    max_gauss = tf.convert_to_tensor(max_gauss, dtype=tf.float64)

    # Evaluation points for numerical integration over a Gaussian.
    z1 = tf.reshape(tf.linspace(-max_gauss, max_gauss, n_gauss), (-1, 1, 1))
    z2 = tf.transpose(z1, perm=[1, 0, 2])

    if log_spacing:
      var_aa = tf.exp(tf.linspace(tf.log(min_var), tf.log(max_var), n_var))
    else:
      # Evaluation points for pre-activations variance and correlation
      var_aa = tf.linspace(min_var, max_var, n_var)
    corr_ab = tf.reshape(tf.linspace(-max_corr, max_corr, n_corr), (1, 1, -1))

    # compute q_aa
    log_weights_aa_unnorm = -0.5 * (z1**2 / tf.reshape(var_aa, [1, 1, -1]))
    log_weights_aa = log_weights_aa_unnorm - tf.reduce_logsumexp(
        log_weights_aa_unnorm, axis=[0, 1], keep_dims=True)
    weights_aa = tf.exp(log_weights_aa)
    qaa = tf.reduce_sum(nonlin_fn(z1)**2 * weights_aa, axis=[0, 1])

    # compute q_ab
    # weights to reweight uniform samples by, for q_ab.
    # (weights are probability of z1, z2 under Gaussian
    #  w/ variance var_aa and covariance var_aa*corr_ab)
    # weights_ab will have shape [n_g, n_g, n_v, n_c]
    def fill_qab_slice(idx):
      return _fill_qab_slice(idx, z1, z2, var_aa, corr_ab, nonlin_fn)

    qab = tf.map_fn(
        fill_qab_slice,
        tf.range(n_var),
        dtype=tf.float64,
        parallel_iterations=multiprocessing.cpu_count())

    var_grid_pts = tf.reshape(var_aa, [-1])
    corr_grid_pts = tf.reshape(corr_ab, [-1])

    return var_grid_pts, corr_grid_pts, qaa, qab

