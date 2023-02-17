import numpy as np
import src.framework as fw
from tensorflow.python.training import moving_averages


def drop_path(x, keep_prob):
  """Drops out a whole example hiddenstate with the specified probability."""

  batch_size = fw.shape(x)[0]
  noise_shape = [batch_size, 1, 1, 1]
  random_tensor = keep_prob
  random_tensor += fw.random_uniform(noise_shape, dtype=fw.float32)
  binary_tensor = fw.floor(random_tensor)
  x = fw.divide(x, keep_prob) * binary_tensor

  return x


def global_avg_pool(x, data_format="NHWC"):
  if data_format == "NHWC":
    x = fw.reduce_mean(x, [1, 2])
  elif data_format == "NCHW":
    x = fw.reduce_mean(x, [2, 3])
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

  with fw.variable_scope(name, reuse=None if is_training else True):
    offset = fw.create_weight(
      "offset", shape,
      initializer=fw.Constant(0.0))
    scale = fw.create_weight(
      "scale", shape,
      initializer=fw.Constant(1.0))
    moving_mean = fw.create_weight(
      "moving_mean", shape, trainable=False,
      initializer=fw.Constant(0.0))
    moving_variance = fw.create_weight(
      "moving_variance", shape, trainable=False,
      initializer=fw.Constant(1.0))

    if is_training:
      x, mean, variance = fw.fused_batch_norm(
        x, scale, offset, epsilon=epsilon, data_format=data_format,
        is_training=True)
      update_mean = moving_averages.assign_moving_average(
        moving_mean, mean, decay)
      update_variance = moving_averages.assign_moving_average(
        moving_variance, variance, decay)
      with fw.control_dependencies([update_mean, update_variance]):
        x = fw.identity(x)
    else:
      x, _, _ = fw.fused_batch_norm(x, scale, offset, mean=moving_mean,
                                       variance=moving_variance,
                                       epsilon=epsilon, data_format=data_format,
                                       is_training=False)
  return x


def batch_norm_with_mask(x, is_training, mask, num_channels, name="bn",
                         decay=0.9, epsilon=1e-3, data_format="NHWC"):

  shape = [num_channels]
  indices = fw.where(mask)
  indices = fw.to_int32(indices)
  indices = fw.reshape(indices, [-1])

  with fw.variable_scope(name, reuse=None if is_training else True):
    offset = fw.create_weight(
      "offset", shape,
      initializer=fw.Constant(0.0))
    scale = fw.create_weight(
      "scale", shape,
      initializer=fw.Constant(1.0))
    offset = fw.boolean_mask(offset, mask)
    scale = fw.boolean_mask(scale, mask)

    moving_mean = fw.create_weight(
      "moving_mean", shape, trainable=False,
      initializer=fw.Constant(0.0))
    moving_variance = fw.create_weight(
      "moving_variance", shape, trainable=False,
      initializer=fw.Constant(1.0))

    if is_training:
      x, mean, variance = fw.fused_batch_norm(
        x, scale, offset, epsilon=epsilon, data_format=data_format,
        is_training=True)
      mean = (1.0 - decay) * (fw.boolean_mask(moving_mean, mask) - mean)
      variance = (1.0 - decay) * (fw.boolean_mask(moving_variance, mask) - variance)
      update_mean = fw.scatter_sub(moving_mean, indices, mean)
      update_variance = fw.scatter_sub(moving_variance, indices, variance)
      with fw.control_dependencies([update_mean, update_variance]):
        x = fw.identity(x)
    else:
      masked_moving_mean = fw.boolean_mask(moving_mean, mask)
      masked_moving_variance = fw.boolean_mask(moving_variance, mask)
      x, _, _ = fw.fused_batch_norm(x, scale, offset,
                                       mean=masked_moving_mean,
                                       variance=masked_moving_variance,
                                       epsilon=epsilon, data_format=data_format,
                                       is_training=False)
  return x
