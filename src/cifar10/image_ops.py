import numpy as np
import src.framework as fw
from tensorflow.python.training import moving_averages


def drop_path(x, keep_prob):
  """Drops out a whole example hiddenstate with the specified probability."""
  return fw.divide(x, keep_prob) * fw.floor(keep_prob + fw.random_uniform(
    [fw.shape(x)[0], 1, 1, 1],
    dtype=fw.float32))

def batch_norm(x, is_training, data_format, weights, name="bn", decay=0.9, epsilon=1e-5):
  shape = [data_format.get_C(x)]

  with fw.name_scope(name) as scope:
    reuse = None if is_training else True
    offset = weights.get(reuse, scope, "offset", shape, fw.constant_initializer(0.0))
    scale = weights.get(reuse, scope, "scale", shape, fw.constant_initializer(1.0))
    moving_mean = weights.get(reuse, scope, "moving_mean", shape, fw.constant_initializer(0.0), trainable=False)
    moving_variance = weights.get(reuse, scope, "moving_variance", shape, fw.constant_initializer(1.0), trainable=False)

    if is_training:
      x, mean, variance = fw.fused_batch_norm(
        x, scale, offset, epsilon=epsilon, data_format=data_format.name,
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
                                       epsilon=epsilon, data_format=data_format.name,
                                       is_training=False)
  return x


def batch_norm_with_mask(x, is_training: bool, mask, num_channels, weights, name="bn",
                         decay=0.9, epsilon=1e-3, data_format="NHWC"):

  shape = [num_channels]
  indices = fw.reshape(fw.to_int32(fw.where(mask)), [-1])

  with fw.name_scope(name) as scope:
    reuse = None if is_training else True
    offset = fw.boolean_mask(
      weights.get(reuse, scope, "offset", shape, fw.constant_initializer(0.0)),
      mask)
    scale = fw.boolean_mask(
      weights.get(reuse, scope, "scale", shape, fw.constant_initializer(1.0)),
      mask)

    moving_mean = weights.get(
      reuse,
      scope,
      "moving_mean",
      shape, fw.constant_initializer(0.0),
      trainable=False)
    moving_variance = weights.get(
      reuse,
      scope,
      "moving_variance",
      shape,
      fw.constant_initializer(1.0),
      trainable=False)

    if is_training:
      x, mean, variance = fw.fused_batch_norm(
        x, scale, offset, epsilon=epsilon, data_format=data_format,
        is_training=True)
      with fw.control_dependencies([
          fw.scatter_sub(moving_mean, indices, (1.0 - decay) * (fw.boolean_mask(moving_mean, mask) - mean)),
          fw.scatter_sub(moving_variance, indices, (1.0 - decay) * (fw.boolean_mask(moving_variance, mask) - variance))]):
        x = fw.identity(x)
    else:
      x, _, _ = fw.fused_batch_norm(x, scale, offset,
                                       mean=fw.boolean_mask(moving_mean, mask),
                                       variance=fw.boolean_mask(moving_variance, mask),
                                       epsilon=epsilon, data_format=data_format,
                                       is_training=False)
  return x
