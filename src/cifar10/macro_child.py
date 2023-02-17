from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import src.framework as fw

from src.cifar10.models import Model
from src.cifar10.image_ops import batch_norm
from src.cifar10.image_ops import batch_norm_with_mask
from src.cifar10.image_ops import global_avg_pool

from src.utils import count_model_params
from src.utils import get_train_ops


class MacroChild(Model):
  def __init__(self,
               images,
               labels,
               cutout_size=None,
               whole_channels=False,
               fixed_arc=None,
               out_filters_scale=1,
               num_layers=2,
               num_branches=6,
               out_filters=24,
               keep_prob=1.0,
               batch_size=32,
               clip_mode=None,
               grad_bound=None,
               l2_reg=1e-4,
               lr_init=0.1,
               lr_dec_start=0,
               lr_dec_every=10000,
               lr_dec_rate=0.1,
               lr_cosine=False,
               lr_max=None,
               lr_min=None,
               lr_T_0=None,
               lr_T_mul=None,
               optim_algo=None,
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               data_format="NHWC",
               name="child",
               *args,
               **kwargs
              ):
    """
    """

    super(self.__class__, self).__init__(
      images,
      labels,
      cutout_size=cutout_size,
      batch_size=batch_size,
      clip_mode=clip_mode,
      grad_bound=grad_bound,
      l2_reg=l2_reg,
      lr_init=lr_init,
      lr_dec_start=lr_dec_start,
      lr_dec_every=lr_dec_every,
      lr_dec_rate=lr_dec_rate,
      keep_prob=keep_prob,
      optim_algo=optim_algo,
      sync_replicas=sync_replicas,
      num_aggregate=num_aggregate,
      num_replicas=num_replicas,
      data_format=data_format,
      name=name)

    self.whole_channels = whole_channels
    self.lr_cosine = lr_cosine
    self.lr_max = lr_max
    self.lr_min = lr_min
    self.lr_T_0 = lr_T_0
    self.lr_T_mul = lr_T_mul
    self.out_filters = out_filters * out_filters_scale
    self.num_layers = num_layers

    self.num_branches = num_branches
    self.fixed_arc = fixed_arc
    self.out_filters_scale = out_filters_scale

    pool_distance = self.num_layers // 3
    self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]

  def _get_HW(self, x):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    return x.get_shape()[2]

  def _get_strides(self, stride):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    if self.data_format == "NHWC":
      return [1, stride, stride, 1]
    elif self.data_format == "NCHW":
      return [1, 1, stride, stride]
    else:
      raise ValueError("Unknown data_format '{0}'".format(self.data_format))

  def _factorized_reduction(self, x, out_filters, stride, is_training):
    """Reduces the shape of x without information loss due to striding."""
    assert out_filters % 2 == 0, (
        "Need even number of filters when using this factorized reduction.")
    if stride == 1:
      with fw.variable_scope("path_conv"):
        inp_c = self._get_C(x)
        w = fw.create_weight("w", [1, 1, inp_c, out_filters])
        x = fw.conv2d(x, w, [1, 1, 1, 1], "SAME",
                         data_format=self.data_format)
        x = batch_norm(x, is_training, data_format=self.data_format)
        return x

    stride_spec = self._get_strides(stride)
    # Skip path 1
    path1 = fw.avg_pool(
        x, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format)
    with fw.variable_scope("path1_conv"):
      inp_c = self._get_C(path1)
      w = fw.create_weight("w", [1, 1, inp_c, out_filters // 2])
      path1 = fw.conv2d(path1, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)

    # Skip path 2
    # First pad with 0"s on the right and bottom, then shift the filter to
    # include those 0"s that were added.
    if self.data_format == "NHWC":
      pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
      path2 = fw.pad(x, pad_arr)[:, 1:, 1:, :]
      concat_axis = 3
    else:
      pad_arr = [[0, 0], [0, 0], [0, 1], [0, 1]]
      path2 = fw.pad(x, pad_arr)[:, :, 1:, 1:]
      concat_axis = 1

    path2 = fw.avg_pool(
        path2, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format)
    with fw.variable_scope("path2_conv"):
      inp_c = self._get_C(path2)
      w = fw.create_weight("w", [1, 1, inp_c, out_filters // 2])
      path2 = fw.conv2d(path2, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)

    # Concat and apply BN
    final_path = fw.concat(values=[path1, path2], axis=concat_axis)
    final_path = batch_norm(final_path, is_training,
                            data_format=self.data_format)

    return final_path


  def _get_C(self, x):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    if self.data_format == "NHWC":
      return x.get_shape()[3]
    elif self.data_format == "NCHW":
      return x.get_shape()[1]
    else:
      raise ValueError("Unknown data_format '{0}'".format(self.data_format))

  def _model(self, images, is_training, reuse=False):
    with fw.variable_scope(self.name, reuse=reuse):
      layers = []

      out_filters = self.out_filters
      with fw.variable_scope("stem_conv"):
        w = fw.create_weight("w", [3, 3, 3, out_filters])
        x = fw.conv2d(images, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
        x = batch_norm(x, is_training, data_format=self.data_format)
        layers.append(x)

      if self.whole_channels:
        start_idx = 0
      else:
        start_idx = self.num_branches
      for layer_id in range(self.num_layers):
        with fw.variable_scope("layer_{0}".format(layer_id)):
          if self.fixed_arc is None:
            x = self._enas_layer(layer_id, layers, start_idx, out_filters, is_training)
          else:
            x = self._fixed_layer(layer_id, layers, start_idx, out_filters, is_training)
          layers.append(x)
          if layer_id in self.pool_layers:
            if self.fixed_arc is not None:
              out_filters *= 2
            with fw.variable_scope("pool_at_{0}".format(layer_id)):
              pooled_layers = []
              for i, layer in enumerate(layers):
                with fw.variable_scope("from_{0}".format(i)):
                  x = self._factorized_reduction(
                    layer, out_filters, 2, is_training)
                pooled_layers.append(x)
              layers = pooled_layers
        if self.whole_channels:
          start_idx += 1 + layer_id
        else:
          start_idx += 2 * self.num_branches + layer_id
        print(layers[-1])

      x = global_avg_pool(x, data_format=self.data_format)
      if is_training:
        x = fw.dropout(x, self.keep_prob)
      with fw.variable_scope("fc"):
        if self.data_format == "NWHC":
          inp_c = x.get_shape()[3]
        elif self.data_format == "NCHW":
          inp_c = x.get_shape()[1]
        else:
          raise ValueError("Unknown data_format {0}".format(self.data_format))
        w = fw.create_weight("w", [inp_c, 10])
        x = fw.matmul(x, w)
    return x

  def _enas_layer(self, layer_id, prev_layers, start_idx, out_filters, is_training):
    """
    Args:
      layer_id: current layer
      prev_layers: cache of previous layers. for skip connections
      start_idx: where to start looking at. technically, we can infer this
        from layer_id, but why bother...
      is_training: for batch_norm
    """

    inputs = prev_layers[-1]
    if self.whole_channels:
      if self.data_format == "NHWC":
        inp_h = inputs.get_shape()[1]
        inp_w = inputs.get_shape()[2]
        inp_c = inputs.get_shape()[3]
      elif self.data_format == "NCHW":
        inp_c = inputs.get_shape()[1]
        inp_h = inputs.get_shape()[2]
        inp_w = inputs.get_shape()[3]

      count = self.sample_arc[start_idx]
      branches = {}
      with fw.variable_scope("branch_0"):
        y = self._conv_branch(inputs, 3, is_training, out_filters, out_filters,
                              start_idx=0)
        branches[fw.equal(count, 0)] = lambda: y
      with fw.variable_scope("branch_1"):
        y = self._conv_branch(inputs, 3, is_training, out_filters, out_filters,
                              start_idx=0, separable=True)
        branches[fw.equal(count, 1)] = lambda: y
      with fw.variable_scope("branch_2"):
        y = self._conv_branch(inputs, 5, is_training, out_filters, out_filters,
                              start_idx=0)
        branches[fw.equal(count, 2)] = lambda: y
      with fw.variable_scope("branch_3"):
        y = self._conv_branch(inputs, 5, is_training, out_filters, out_filters,
                              start_idx=0, separable=True)
        branches[fw.equal(count, 3)] = lambda: y
      if self.num_branches >= 5:
        with fw.variable_scope("branch_4"):
          y = self._pool_branch(inputs, is_training, out_filters, "avg",
                                start_idx=0)
        branches[fw.equal(count, 4)] = lambda: y
      if self.num_branches >= 6:
        with fw.variable_scope("branch_5"):
          y = self._pool_branch(inputs, is_training, out_filters, "max",
                                start_idx=0)
        branches[fw.equal(count, 5)] = lambda: y
      out = fw.case(branches, default=lambda: fw.constant(0, fw.float32),
                    exclusive=True)

      if self.data_format == "NHWC":
        out.set_shape([None, inp_h, inp_w, out_filters])
      elif self.data_format == "NCHW":
        out.set_shape([None, out_filters, inp_h, inp_w])
    else:
      count = self.sample_arc[start_idx:start_idx + 2 * self.num_branches]
      branches = []
      with fw.variable_scope("branch_0"):
        branches.append(self._conv_branch(inputs, 3, is_training, count[1],
                                          out_filters, start_idx=count[0]))
      with fw.variable_scope("branch_1"):
        branches.append(self._conv_branch(inputs, 3, is_training, count[3],
                                          out_filters, start_idx=count[2],
                                          separable=True))
      with fw.variable_scope("branch_2"):
        branches.append(self._conv_branch(inputs, 5, is_training, count[5],
                                          out_filters, start_idx=count[4]))
      with fw.variable_scope("branch_3"):
        branches.append(self._conv_branch(inputs, 5, is_training, count[7],
                                          out_filters, start_idx=count[6],
                                          separable=True))
      if self.num_branches >= 5:
        with fw.variable_scope("branch_4"):
          branches.append(self._pool_branch(inputs, is_training, count[9],
                                            "avg", start_idx=count[8]))
      if self.num_branches >= 6:
        with fw.variable_scope("branch_5"):
          branches.append(self._pool_branch(inputs, is_training, count[11],
                                            "max", start_idx=count[10]))

      with fw.variable_scope("final_conv"):
        w = fw.create_weight("w", [self.num_branches * out_filters, out_filters])
        w_mask = fw.constant([False] * (self.num_branches * out_filters), fw.bool)
        new_range = fw.range(0, self.num_branches * out_filters, dtype=fw.int32)
        for i in range(self.num_branches):
          start = out_filters * i + count[2 * i]
          new_mask = fw.logical_and(
            start <= new_range, new_range < start + count[2 * i + 1])
          w_mask = fw.logical_or(w_mask, new_mask)
        w = fw.boolean_mask(w, w_mask)
        w = fw.reshape(w, [1, 1, -1, out_filters])

        inp = prev_layers[-1]
        if self.data_format == "NHWC":
          branches = fw.concat(branches, axis=3)
        elif self.data_format == "NCHW":
          branches = fw.concat(branches, axis=1)
          N = fw.shape(inp)[0]
          H = inp.get_shape()[2]
          W = inp.get_shape()[3]
          branches = fw.reshape(branches, [N, -1, H, W])
        out = fw.conv2d(
          branches, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
        out = batch_norm(out, is_training, data_format=self.data_format)
        out = fw.relu(out)

    if layer_id > 0:
      if self.whole_channels:
        skip_start = start_idx + 1
      else:
        skip_start = start_idx + 2 * self.num_branches
      skip = self.sample_arc[skip_start: skip_start + layer_id]
      with fw.variable_scope("skip"):
        res_layers = []
        for i in range(layer_id):
          res_layers.append(fw.cond(fw.equal(skip[i], 1),
                                    lambda: prev_layers[i],
                                    lambda: fw.zeros_like(prev_layers[i])))
        res_layers.append(out)
        out = fw.add_n(res_layers)
        out = batch_norm(out, is_training, data_format=self.data_format)

    return out

  def _fixed_layer(
      self, layer_id, prev_layers, start_idx, out_filters, is_training):
    """
    Args:
      layer_id: current layer
      prev_layers: cache of previous layers. for skip connections
      start_idx: where to start looking at. technically, we can infer this
        from layer_id, but why bother...
      is_training: for batch_norm
    """

    inputs = prev_layers[-1]
    if self.whole_channels:
      if self.data_format == "NHWC":
        inp_c = inputs.get_shape()[3]
      elif self.data_format == "NCHW":
        inp_c = inputs.get_shape()[1]

      count = self.sample_arc[start_idx]
      if count in [0, 1, 2, 3]:
        size = [3, 3, 5, 5]
        filter_size = size[count]
        with fw.variable_scope("conv_1x1"):
          w = fw.create_weight("w", [1, 1, inp_c, out_filters])
          out = fw.relu(inputs)
          out = fw.conv2d(out, w, [1, 1, 1, 1], "SAME",
                             data_format=self.data_format)
          out = batch_norm(out, is_training, data_format=self.data_format)

        with fw.variable_scope("conv_{0}x{0}".format(filter_size)):
          w = fw.create_weight("w", [filter_size, filter_size, out_filters, out_filters])
          out = fw.relu(out)
          out = fw.conv2d(out, w, [1, 1, 1, 1], "SAME",
                             data_format=self.data_format)
          out = batch_norm(out, is_training, data_format=self.data_format)
      elif count == 4:
        pass
      elif count == 5:
        pass
      else:
        raise ValueError("Unknown operation number '{0}'".format(count))
    else:
      count = (self.sample_arc[start_idx:start_idx + 2*self.num_branches] *
               self.out_filters_scale)
      branches = []
      total_out_channels = 0
      with fw.variable_scope("branch_0"):
        total_out_channels += count[1]
        branches.append(self._conv_branch(inputs, 3, is_training, count[1]))
      with fw.variable_scope("branch_1"):
        total_out_channels += count[3]
        branches.append(
          self._conv_branch(inputs, 3, is_training, count[3], separable=True))
      with fw.variable_scope("branch_2"):
        total_out_channels += count[5]
        branches.append(self._conv_branch(inputs, 5, is_training, count[5]))
      with fw.variable_scope("branch_3"):
        total_out_channels += count[7]
        branches.append(
          self._conv_branch(inputs, 5, is_training, count[7], separable=True))
      if self.num_branches >= 5:
        with fw.variable_scope("branch_4"):
          total_out_channels += count[9]
          branches.append(
            self._pool_branch(inputs, is_training, count[9], "avg"))
      if self.num_branches >= 6:
        with fw.variable_scope("branch_5"):
          total_out_channels += count[11]
          branches.append(
            self._pool_branch(inputs, is_training, count[11], "max"))

      with fw.variable_scope("final_conv"):
        w = fw.create_weight("w", [1, 1, total_out_channels, out_filters])
        if self.data_format == "NHWC":
          branches = fw.concat(branches, axis=3)
        elif self.data_format == "NCHW":
          branches = fw.concat(branches, axis=1)
        out = fw.relu(branches)
        out = fw.conv2d(out, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)
        out = batch_norm(out, is_training, data_format=self.data_format)

    if layer_id > 0:
      if self.whole_channels:
        skip_start = start_idx + 1
      else:
        skip_start = start_idx + 2 * self.num_branches
      skip = self.sample_arc[skip_start: skip_start + layer_id]
      total_skip_channels = np.sum(skip) + 1

      res_layers = []
      for i in range(layer_id):
        if skip[i] == 1:
          res_layers.append(prev_layers[i])
      prev = res_layers + [out]

      if self.data_format == "NHWC":
        prev = fw.concat(prev, axis=3)
      elif self.data_format == "NCHW":
        prev = fw.concat(prev, axis=1)

      out = prev
      with fw.variable_scope("skip"):
        w = fw.create_weight(
          "w", [1, 1, total_skip_channels * out_filters, out_filters])
        out = fw.relu(out)
        out = fw.conv2d(
          out, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
        out = batch_norm(out, is_training, data_format=self.data_format)

    return out

  def _conv_branch(self, inputs, filter_size, is_training, count, out_filters,
                   ch_mul=1, start_idx=None, separable=False):
    """
    Args:
      start_idx: where to start taking the output channels. if None, assuming
        fixed_arc mode
      count: how many output_channels to take.
    """

    if start_idx is None:
      assert self.fixed_arc is not None, "you screwed up!"

    if self.data_format == "NHWC":
      inp_c = inputs.get_shape()[3]
    elif self.data_format == "NCHW":
      inp_c = inputs.get_shape()[1]

    with fw.variable_scope("inp_conv_1"):
      w = fw.create_weight("w", [1, 1, inp_c, out_filters])
      x = fw.conv2d(inputs, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
      x = batch_norm(x, is_training, data_format=self.data_format)
      x = fw.relu(x)

    with fw.variable_scope("out_conv_{}".format(filter_size)):
      if start_idx is None:
        if separable:
          w_depth = fw.create_weight(
            "w_depth", [self.filter_size, self.filter_size, out_filters, ch_mul])
          w_point = fw.create_weight("w_point", [1, 1, out_filters * ch_mul, count])
          x = fw.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1],
                                     padding="SAME", data_format=self.data_format)
          x = batch_norm(x, is_training, data_format=self.data_format)
        else:
          w = fw.create_weight("w", [filter_size, filter_size, inp_c, count])
          x = fw.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
          x = batch_norm(x, is_training, data_format=self.data_format)
      else:
        if separable:
          w_depth = fw.create_weight("w_depth", [filter_size, filter_size, out_filters, ch_mul])
          w_point = fw.create_weight("w_point", [out_filters, out_filters * ch_mul])
          w_point = w_point[start_idx:start_idx+count, :]
          w_point = fw.transpose(w_point, [1, 0])
          w_point = fw.reshape(w_point, [1, 1, out_filters * ch_mul, count])

          x = fw.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1],
                                     padding="SAME", data_format=self.data_format)
          mask = fw.range(0, out_filters, dtype=fw.int32)
          mask = fw.logical_and(start_idx <= mask, mask < start_idx + count)
          x = batch_norm_with_mask(
            x, is_training, mask, out_filters, data_format=self.data_format)
        else:
          w = fw.create_weight("w", [filter_size, filter_size, out_filters, out_filters])
          w = fw.transpose(w, [3, 0, 1, 2])
          w = w[start_idx:start_idx+count, :, :, :]
          w = fw.transpose(w, [1, 2, 3, 0])
          x = fw.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
          mask = fw.range(0, out_filters, dtype=fw.int32)
          mask = fw.logical_and(start_idx <= mask, mask < start_idx + count)
          x = batch_norm_with_mask(
            x, is_training, mask, out_filters, data_format=self.data_format)
      x = fw.relu(x)
    return x

  def _pool_branch(self, inputs, is_training, count, avg_or_max, start_idx=None):
    """
    Args:
      start_idx: where to start taking the output channels. if None, assuming
        fixed_arc mode
      count: how many output_channels to take.
    """

    if start_idx is None:
      assert self.fixed_arc is not None, "you screwed up!"

    if self.data_format == "NHWC":
      inp_c = inputs.get_shape()[3]
    elif self.data_format == "NCHW":
      inp_c = inputs.get_shape()[1]

    with fw.variable_scope("conv_1"):
      w = fw.create_weight("w", [1, 1, inp_c, self.out_filters])
      x = fw.conv2d(inputs, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
      x = batch_norm(x, is_training, data_format=self.data_format)
      x = fw.relu(x)

    with fw.variable_scope("pool"):
      if self.data_format == "NHWC":
        actual_data_format = "channels_last"
      elif self.data_format == "NCHW":
        actual_data_format = "channels_first"

      if avg_or_max == "avg":
        x = fw.avg_pool2d(
          x, [3, 3], [1, 1], "SAME", data_format=actual_data_format)
      elif avg_or_max == "max":
        x = fw.max_pool2d(
          x, [3, 3], [1, 1], "SAME", data_format=actual_data_format)
      else:
        raise ValueError("Unknown pool {}".format(avg_or_max))

      if start_idx is not None:
        if self.data_format == "NHWC":
          x = x[:, :, :, start_idx : start_idx+count]
        elif self.data_format == "NCHW":
          x = x[:, start_idx : start_idx+count, :, :]

    return x

  # override
  def _build_train(self):
    print("-" * 80)
    print("Build train graph")
    logits = self._model(self.x_train, is_training=True)
    log_probs = fw.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=self.y_train)
    self.loss = fw.reduce_mean(log_probs)

    self.train_preds = fw.argmax(logits, axis=1)
    self.train_preds = fw.to_int32(self.train_preds)
    self.train_acc = fw.equal(self.train_preds, self.y_train)
    self.train_acc = fw.to_int32(self.train_acc)
    self.train_acc = fw.reduce_sum(self.train_acc)

    tf_variables = [var
        for var in fw.trainable_variables() if var.name.startswith(self.name)]
    self.num_vars = count_model_params(tf_variables)
    print("Model has {} params".format(self.num_vars))

    self.global_step = fw.get_or_create_global_step()
    self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
      self.loss,
      tf_variables,
      self.global_step,
      clip_mode=self.clip_mode,
      grad_bound=self.grad_bound,
      l2_reg=self.l2_reg,
      lr_init=self.lr_init,
      lr_dec_start=self.lr_dec_start,
      lr_dec_every=self.lr_dec_every,
      lr_dec_rate=self.lr_dec_rate,
      lr_cosine=self.lr_cosine,
      lr_max=self.lr_max,
      lr_min=self.lr_min,
      lr_T_0=self.lr_T_0,
      lr_T_mul=self.lr_T_mul,
      num_train_batches=self.num_train_batches,
      optim_algo=self.optim_algo,
      sync_replicas=self.sync_replicas,
      num_aggregate=self.num_aggregate,
      num_replicas=self.num_replicas)

  # override
  def _build_valid(self):
    if self.x_valid is not None:
      print("-" * 80)
      print("Build valid graph")
      logits = self._model(self.x_valid, False, reuse=True)
      self.valid_preds = fw.argmax(logits, axis=1)
      self.valid_preds = fw.to_int32(self.valid_preds)
      self.valid_acc = fw.equal(self.valid_preds, self.y_valid)
      self.valid_acc = fw.to_int32(self.valid_acc)
      self.valid_acc = fw.reduce_sum(self.valid_acc)

  # override
  def _build_test(self):
    print("-" * 80)
    print("Build test graph")
    logits = self._model(self.x_test, False, reuse=True)
    self.test_preds = fw.argmax(logits, axis=1)
    self.test_preds = fw.to_int32(self.test_preds)
    self.test_acc = fw.equal(self.test_preds, self.y_test)
    self.test_acc = fw.to_int32(self.test_acc)
    self.test_acc = fw.reduce_sum(self.test_acc)

  # override
  def build_valid_rl(self, shuffle=False):
    print("-" * 80)
    print("Build valid graph on shuffled data")
    with fw.device("/gpu:0"):
      # shuffled valid data: for choosing validation model
      if not shuffle and self.data_format == "NCHW":
        self.images["valid_original"] = np.transpose(
          self.images["valid_original"], [0, 3, 1, 2])
      x_valid_shuffle, y_valid_shuffle = fw.shuffle_batch(
        [self.images["valid_original"], self.labels["valid_original"]],
        self.batch_size,
        self.seed)

      def _pre_process(x):
        x = fw.pad(x, [[4, 4], [4, 4], [0, 0]])
        x = fw.image.random_crop(x, [32, 32, 3], seed=self.seed)
        x = fw.image.random_flip_left_right(x, seed=self.seed)
        if self.data_format == "NCHW":
          x = fw.transpose(x, [2, 0, 1])

        return x

      if shuffle:
        x_valid_shuffle = fw.map_fn(
          _pre_process, x_valid_shuffle, back_prop=False)

    logits = self._model(x_valid_shuffle, False, reuse=True)
    valid_shuffle_preds = fw.argmax(logits, axis=1)
    valid_shuffle_preds = fw.to_int32(valid_shuffle_preds)
    self.valid_shuffle_acc = fw.equal(valid_shuffle_preds, y_valid_shuffle)
    self.valid_shuffle_acc = fw.to_int32(self.valid_shuffle_acc)
    self.valid_shuffle_acc = fw.reduce_sum(self.valid_shuffle_acc)

  def connect_controller(self, controller_model):
    if self.fixed_arc is None:
      self.sample_arc = controller_model.sample_arc
    else:
      fixed_arc = np.array([int(x) for x in self.fixed_arc.split(" ") if x])
      self.sample_arc = fixed_arc

    self._build_train()
    self._build_valid()
    self._build_test()

