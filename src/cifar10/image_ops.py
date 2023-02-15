import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

from src.common_ops import create_weight


def drop_path(x, keep_prob):
  """Drops out a whole example hiddenstate with the specified probability."""

  batch_size = tf.shape(x)[0]
  noise_shape = [batch_size, 1, 1, 1]
  random_tensor = keep_prob
  random_tensor += tf.random.uniform(noise_shape, dtype=tf.float32)
  binary_tensor = tf.floor(random_tensor)
  x = tf.math.divide(x, keep_prob) * binary_tensor

  return x


def global_avg_pool(x, data_format="NHWC"):
  if data_format == "NHWC":
    x = tf.reduce_mean(x, [1, 2])
  elif data_format == "NCHW":
    x = tf.reduce_mean(x, [2, 3])
  else:
    raise NotImplementedError("Unknown data_format {}".format(data_format))
  return x


def batch_norm(x, is_training, name="bn", decay=0.9, epsilon=1e-5,
               data_format="NHWC"):
  if data_format == "NHWC":
    shape = [x.get_shape()[3]]
  elif data_format == "NCHW":
    shape = [x.get_shape()[1]]
  else:
    raise NotImplementedError("Unknown data_format {}".format(data_format))

  with tf.compat.v1.variable_scope(name, reuse=None if is_training else True):
    offset = tf.compat.v1.get_variable(
      "offset", shape,
      initializer=tf.compat.v1.keras.initializers.Constant(0.0, dtype=tf.float32))
    scale = tf.compat.v1.get_variable(
      "scale", shape,
      initializer=tf.compat.v1.keras.initializers.Constant(1.0, dtype=tf.float32))
    moving_mean = tf.compat.v1.get_variable(
      "moving_mean", shape, trainable=False,
      initializer=tf.compat.v1.keras.initializers.Constant(0.0, dtype=tf.float32))
    moving_variance = tf.compat.v1.get_variable(
      "moving_variance", shape, trainable=False,
      initializer=tf.compat.v1.keras.initializers.Constant(1.0, dtype=tf.float32))

    if is_training:
      x, mean, variance = tf.compat.v1.nn.fused_batch_norm(
        x, scale, offset, epsilon=epsilon, data_format=data_format,
        is_training=True)
      update_mean = moving_averages.assign_moving_average(
        moving_mean, mean, decay)
      update_variance = moving_averages.assign_moving_average(
        moving_variance, variance, decay)
      with tf.control_dependencies([update_mean, update_variance]):
        x = tf.identity(x)
    else:
      x, _, _ = tf.compat.v1.nn.fused_batch_norm(x, scale, offset, mean=moving_mean,
                                       variance=moving_variance,
                                       epsilon=epsilon, data_format=data_format,
                                       is_training=False)
  return x


def batch_norm_with_mask(x, is_training, mask, num_channels, name="bn",
                         decay=0.9, epsilon=1e-3, data_format="NHWC"):

  shape = [num_channels]
  indices = tf.where(mask)
  indices = tf.compat.v1.to_int32(indices)
  indices = tf.reshape(indices, [-1])

  with tf.compat.v1.variable_scope(name, reuse=None if is_training else True):
    offset = tf.compat.v1.get_variable(
      "offset", shape,
      initializer=tf.compat.v1.keras.initializers.Constant(0.0, dtype=tf.float32))
    scale = tf.compat.v1.get_variable(
      "scale", shape,
      initializer=tf.compat.v1.keras.initializers.Constant(1.0, dtype=tf.float32))
    offset = tf.boolean_mask(offset, mask)
    scale = tf.boolean_mask(scale, mask)

    moving_mean = tf.compat.v1.get_variable(
      "moving_mean", shape, trainable=False,
      initializer=tf.compat.v1.keras.initializers.Constant(0.0, dtype=tf.float32))
    moving_variance = tf.compat.v1.get_variable(
      "moving_variance", shape, trainable=False,
      initializer=tf.compat.v1.keras.initializers.Constant(1.0, dtype=tf.float32))

    if is_training:
      x, mean, variance = tf.compat.v1.nn.fused_batch_norm(
        x, scale, offset, epsilon=epsilon, data_format=data_format,
        is_training=True)
      mean = (1.0 - decay) * (tf.boolean_mask(moving_mean, mask) - mean)
      variance = (1.0 - decay) * (tf.boolean_mask(moving_variance, mask) - variance)
      update_mean = tf.compat.v1.scatter_sub(moving_mean, indices, mean, use_locking=True)
      update_variance = tf.compat.v1.scatter_sub(
        moving_variance, indices, variance, use_locking=True)
      with tf.control_dependencies([update_mean, update_variance]):
        x = tf.identity(x)
    else:
      masked_moving_mean = tf.boolean_mask(moving_mean, mask)
      masked_moving_variance = tf.boolean_mask(moving_variance, mask)
      x, _, _ = tf.compat.v1.nn.fused_batch_norm(x, scale, offset,
                                       mean=masked_moving_mean,
                                       variance=masked_moving_variance,
                                       epsilon=epsilon, data_format=data_format,
                                       is_training=False)
  return x
