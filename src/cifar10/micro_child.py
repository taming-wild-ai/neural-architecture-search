from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import src.framework as fw

from src.cifar10.child import Child
from src.cifar10.image_ops import batch_norm
from src.cifar10.image_ops import drop_path
from src.cifar10.image_ops import global_avg_pool

from src.utils import count_model_params
from src.utils import get_train_ops
from src.utils import DEFINE_boolean, DEFINE_float, DEFINE_integer

DEFINE_float("child_drop_path_keep_prob", 1.0, "minimum drop_path_keep_prob")
DEFINE_integer("child_num_cells", 5, "")
DEFINE_boolean("child_use_aux_heads", False, "Should we use an aux head")
DEFINE_integer("num_epochs", 310, "")

class MicroChild(Child):
  def __init__(self,
               images,
               labels,
               clip_mode=None,
               lr_dec_start=0,
               lr_min=None,
               optim_algo=None,
               name="child",
               **kwargs
              ):
    """
    """

    super(self.__class__, self).__init__(
      images,
      labels,
      clip_mode=clip_mode,
      lr_dec_start=lr_dec_start,
      optim_algo=optim_algo,
      name=name)
    FLAGS = fw.FLAGS
    if self.data_format == "NHWC":
      self.actual_data_format = "channels_last"
    elif self.data_format == "NCHW":
      self.actual_data_format = "channels_first"
    else:
      raise ValueError("Unknown data_format '{0}'".format(self.data_format))

    self.use_aux_heads = FLAGS.child_use_aux_heads
    self.num_epochs = FLAGS.num_epochs
    self.num_train_steps = self.num_epochs * self.num_train_batches
    self.drop_path_keep_prob = FLAGS.child_drop_path_keep_prob
    self.lr_min = lr_min
    self.num_cells = FLAGS.child_num_cells

    self.global_step = fw.get_or_create_global_step()

    if self.drop_path_keep_prob is not None:
      assert self.num_epochs is not None, "Need num_epochs to drop_path"

    pool_distance = self.num_layers // 3
    self.pool_layers = [pool_distance, 2 * pool_distance + 1]

    if self.use_aux_heads:
      self.aux_head_indices = [self.pool_layers[-1] + 1]

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
      path1 = fw.conv2d(path1, w, [1, 1, 1, 1], "VALID",
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
      path2 = fw.conv2d(path2, w, [1, 1, 1, 1], "VALID",
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

  def _apply_drop_path(self, x, layer_id):
    drop_path_keep_prob = self.drop_path_keep_prob

    layer_ratio = float(layer_id + 1) / (self.num_layers + 2)
    drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)

    step_ratio = fw.to_float(self.global_step + 1) / fw.to_float(self.num_train_steps)
    step_ratio = fw.minimum(1.0, step_ratio)
    drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)

    x = drop_path(x, drop_path_keep_prob)
    return x

  def _maybe_calibrate_size(self, layers, out_filters, is_training):
    """Makes sure layers[0] and layers[1] have the same shapes."""

    hw = [self._get_HW(layer) for layer in layers]
    c = [self._get_C(layer) for layer in layers]

    with fw.variable_scope("calibrate"):
      x = layers[0]
      if hw[0] != hw[1]:
        assert hw[0] == 2 * hw[1]
        with fw.variable_scope("pool_x"):
          x = fw.relu(x)
          x = self._factorized_reduction(x, out_filters, 2, is_training)
      elif c[0] != out_filters:
        with fw.variable_scope("pool_x"):
          w = fw.create_weight("w", [1, 1, c[0], out_filters])
          x = fw.relu(x)
          x = fw.conv2d(x, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)
          x = batch_norm(x, is_training, data_format=self.data_format)

      y = layers[1]
      if c[1] != out_filters:
        with fw.variable_scope("pool_y"):
          w = fw.create_weight("w", [1, 1, c[1], out_filters])
          y = fw.relu(y)
          y = fw.conv2d(y, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)
          y = batch_norm(y, is_training, data_format=self.data_format)
    return [x, y]

  def _model(self, images, is_training, reuse=False):
    """Compute the logits given the images."""

    if self.fixed_arc is None:
      is_training = True

    with fw.variable_scope(self.name, reuse=reuse):
      # the first two inputs
      with fw.variable_scope("stem_conv"):
        w = fw.create_weight("w", [3, 3, 3, self.out_filters * 3])
        x = fw.conv2d(
          images, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
        x = batch_norm(x, is_training, data_format=self.data_format)
      if self.data_format == "NHCW":
        split_axis = 3
      elif self.data_format == "NCHW":
        split_axis = 1
      else:
        raise ValueError("Unknown data_format '{0}'".format(self.data_format))
      layers = [x, x]

      # building layers in the micro space
      out_filters = self.out_filters
      for layer_id in range(self.num_layers + 2):
        with fw.variable_scope("layer_{0}".format(layer_id)):
          if layer_id not in self.pool_layers:
            if self.fixed_arc is None:
              x = self._enas_layer(
                layer_id, layers, self.normal_arc, out_filters)
            else:
              x = self._fixed_layer(
                layer_id, layers, self.normal_arc, out_filters, 1, is_training,
                normal_or_reduction_cell="normal")
          else:
            out_filters *= 2
            if self.fixed_arc is None:
              x = self._factorized_reduction(x, out_filters, 2, is_training)
              layers = [layers[-1], x]
              x = self._enas_layer(
                layer_id, layers, self.reduce_arc, out_filters)
            else:
              x = self._fixed_layer(
                layer_id, layers, self.reduce_arc, out_filters, 2, is_training,
                normal_or_reduction_cell="reduction")
          print("Layer {0:>2d}: {1}".format(layer_id, x))
          layers = [layers[-1], x]

        # auxiliary heads
        self.num_aux_vars = 0
        if (self.use_aux_heads and
            layer_id in self.aux_head_indices
            and is_training):
          print("Using aux_head at layer {0}".format(layer_id))
          with fw.variable_scope("aux_head"):
            aux_logits = fw.relu(x)
            aux_logits = fw.avg_pool2d(
              aux_logits, [5, 5], [3, 3], "VALID",
              data_format=self.actual_data_format)
            with fw.variable_scope("proj"):
              inp_c = self._get_C(aux_logits)
              w = fw.create_weight("w", [1, 1, inp_c, 128])
              aux_logits = fw.conv2d(aux_logits, w, [1, 1, 1, 1], "SAME",
                                        data_format=self.data_format)
              aux_logits = batch_norm(aux_logits, is_training=True,
                                      data_format=self.data_format)
              aux_logits = fw.relu(aux_logits)

            with fw.variable_scope("avg_pool"):
              inp_c = self._get_C(aux_logits)
              hw = self._get_HW(aux_logits)
              w = fw.create_weight("w", [hw, hw, inp_c, 768])
              aux_logits = fw.conv2d(aux_logits, w, [1, 1, 1, 1], "SAME",
                                        data_format=self.data_format)
              aux_logits = batch_norm(aux_logits, is_training=True,
                                      data_format=self.data_format)
              aux_logits = fw.relu(aux_logits)

            with fw.variable_scope("fc"):
              aux_logits = global_avg_pool(aux_logits,
                                           data_format=self.data_format)
              inp_c = aux_logits.get_shape()[1]
              w = fw.create_weight("w", [inp_c, 10])
              aux_logits = fw.matmul(aux_logits, w)
              self.aux_logits = aux_logits

          aux_head_variables = [
            var for var in fw.trainable_variables() if (
              var.name.startswith(self.name) and "aux_head" in var.name)]
          self.num_aux_vars = count_model_params(aux_head_variables)
          print("Aux head uses {0} params".format(self.num_aux_vars))

      x = fw.relu(x)
      x = global_avg_pool(x, data_format=self.data_format)
      if is_training and self.keep_prob is not None and self.keep_prob < 1.0:
        x = fw.dropout(x, self.keep_prob)
      with fw.variable_scope("fc"):
        inp_c = self._get_C(x)
        w = fw.create_weight("w", [inp_c, 10])
        x = fw.matmul(x, w)
    return x

  def _fixed_conv(self, x, f_size, out_filters, stride, is_training,
                  stack_convs=2):
    """Apply fixed convolution.

    Args:
      stacked_convs: number of separable convs to apply.
    """

    for conv_id in range(stack_convs):
      inp_c = self._get_C(x)
      if conv_id == 0:
        strides = self._get_strides(stride)
      else:
        strides = [1, 1, 1, 1]

      with fw.variable_scope("sep_conv_{}".format(conv_id)):
        w_depthwise = fw.create_weight("w_depth", [f_size, f_size, inp_c, 1])
        w_pointwise = fw.create_weight("w_point", [1, 1, inp_c, out_filters])
        x = fw.relu(x)
        x = fw.separable_conv2d(
          x,
          depthwise_filter=w_depthwise,
          pointwise_filter=w_pointwise,
          strides=strides, padding="SAME", data_format=self.data_format)
        x = batch_norm(x, is_training, data_format=self.data_format)

    return x

  def _fixed_combine(self, layers, used, out_filters, is_training,
                     normal_or_reduction_cell="normal"):
    """Adjust if necessary.

    Args:
      layers: a list of tf tensors of size [NHWC] of [NCHW].
      used: a numpy tensor, [0] means not used.
    """

    out_hw = min([self._get_HW(layer)
                  for i, layer in enumerate(layers) if used[i] == 0])
    out = []

    with fw.variable_scope("final_combine"):
      for i, layer in enumerate(layers):
        # print(f"*** layer = {layer}")
        if used[i] == 0:
          hw = self._get_HW(layer)
          if hw > out_hw:
            assert hw == out_hw * 2, ("i_hw={0} != {1}=o_hw".format(hw, out_hw))
            with fw.variable_scope("calibrate_{0}".format(i)):
              x = self._factorized_reduction(layer, out_filters, 2, is_training)
          else:
            x = layer
          out.append(x)

      if self.data_format == "NHWC":
        out = fw.concat(out, axis=3)
      elif self.data_format == "NCHW":
        out = fw.concat(out, axis=1)
      else:
        raise ValueError("Unknown data_format '{0}'".format(self.data_format))

    return out

  def _fixed_layer(self, layer_id, prev_layers, arc, out_filters, stride,
                   is_training, normal_or_reduction_cell="normal"):
    """
    Args:
      prev_layers: cache of previous layers. for skip connections
      is_training: for batch_norm
    """

    assert len(prev_layers) == 2
    layers = [prev_layers[0], prev_layers[1]]
    layers = self._maybe_calibrate_size(layers, out_filters,
                                        is_training=is_training)

    with fw.variable_scope("layer_base"):
      x = layers[1]
      inp_c = self._get_C(x)
      w = fw.create_weight("w", [1, 1, inp_c, out_filters])
      x = fw.relu(x)
      x = fw.conv2d(x, w, [1, 1, 1, 1], "SAME",
                       data_format=self.data_format)
      x = batch_norm(x, is_training, data_format=self.data_format)
      layers[1] = x

    used = np.zeros([self.num_cells + 2], dtype=np.int32)
    f_sizes = [3, 5]
    for cell_id in range(self.num_cells):
      with fw.variable_scope("cell_{}".format(cell_id)):
        x_id = arc[4 * cell_id]
        used[x_id] += 1
        x_op = arc[4 * cell_id + 1]
        x = layers[x_id]
        x_stride = stride if x_id in [0, 1] else 1
        with fw.variable_scope("x_conv"):
          if x_op in [0, 1]:
            f_size = f_sizes[x_op]
            x = self._fixed_conv(x, f_size, out_filters, x_stride, is_training)
          elif x_op in [2, 3]:
            inp_c = self._get_C(x)
            if x_op == 2:
              x = fw.avg_pool2d(
                x, [3, 3], [x_stride, x_stride], "SAME",
                data_format=self.actual_data_format)
            else:
              x = fw.max_pool2d(
                x, [3, 3], [x_stride, x_stride], "SAME",
                data_format=self.actual_data_format)
            if inp_c != out_filters:
              w = fw.create_weight("w", [1, 1, inp_c, out_filters])
              x = fw.relu(x)
              x = fw.conv2d(x, w, [1, 1, 1, 1], "SAME",
                               data_format=self.data_format)
              x = batch_norm(x, is_training, data_format=self.data_format)
          else:
            inp_c = self._get_C(x)
            if x_stride > 1:
              assert x_stride == 2
              x = self._factorized_reduction(x, out_filters, 2, is_training)
            if inp_c != out_filters:
              w = fw.create_weight("w", [1, 1, inp_c, out_filters])
              x = fw.relu(x)
              x = fw.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
              x = batch_norm(x, is_training, data_format=self.data_format)
          if (x_op in [0, 1, 2, 3] and
              self.drop_path_keep_prob is not None and
              is_training):
            x = self._apply_drop_path(x, layer_id)

        y_id = arc[4 * cell_id + 2]
        used[y_id] += 1
        y_op = arc[4 * cell_id + 3]
        y = layers[y_id]
        y_stride = stride if y_id in [0, 1] else 1
        with fw.variable_scope("y_conv"):
          if y_op in [0, 1]:
            f_size = f_sizes[y_op]
            y = self._fixed_conv(y, f_size, out_filters, y_stride, is_training)
          elif y_op in [2, 3]:
            inp_c = self._get_C(y)
            if y_op == 2:
              y = fw.avg_pool2d(
                y, [3, 3], [y_stride, y_stride], "SAME",
                data_format=self.actual_data_format)
            else:
              y = fw.max_pool2d(
                y, [3, 3], [y_stride, y_stride], "SAME",
                data_format=self.actual_data_format)
            if inp_c != out_filters:
              w = fw.create_weight("w", [1, 1, inp_c, out_filters])
              y = fw.relu(y)
              y = fw.conv2d(y, w, [1, 1, 1, 1], "SAME",
                               data_format=self.data_format)
              y = batch_norm(y, is_training, data_format=self.data_format)
          else:
            inp_c = self._get_C(y)
            if y_stride > 1:
              assert y_stride == 2
              y = self._factorized_reduction(y, out_filters, 2, is_training)
            if inp_c != out_filters:
              w = fw.create_weight("w", [1, 1, inp_c, out_filters])
              y = fw.relu(y)
              y = fw.conv2d(y, w, [1, 1, 1, 1], "SAME",
                               data_format=self.data_format)
              y = batch_norm(y, is_training, data_format=self.data_format)

          if (y_op in [0, 1, 2, 3] and
              self.drop_path_keep_prob is not None and
              is_training):
            y = self._apply_drop_path(y, layer_id)

        out = x + y
        layers.append(out)
    out = self._fixed_combine(layers, used, out_filters, is_training,
                              normal_or_reduction_cell)

    return out

  def _enas_cell(self, x, curr_cell, prev_cell, op_id, out_filters):
    """Performs an enas operation specified by op_id."""

    num_possible_inputs = curr_cell + 1

    with fw.variable_scope("avg_pool"):
      avg_pool = fw.avg_pool2d(
        x, [3, 3], [1, 1], "SAME", data_format=self.actual_data_format)
      avg_pool_c = self._get_C(avg_pool)
      if avg_pool_c != out_filters:
        with fw.variable_scope("conv"):
          w = fw.create_weight(
            "w", [num_possible_inputs, avg_pool_c * out_filters])
          w = w[prev_cell]
          w = fw.reshape(w, [1, 1, avg_pool_c, out_filters])
          avg_pool = fw.relu(avg_pool)
          avg_pool = fw.conv2d(avg_pool, w, strides=[1, 1, 1, 1],
                                  padding="SAME", data_format=self.data_format)
          avg_pool = batch_norm(avg_pool, is_training=True,
                                data_format=self.data_format)

    with fw.variable_scope("max_pool"):
      max_pool = fw.max_pool2d(
        x, [3, 3], [1, 1], "SAME", data_format=self.actual_data_format)
      max_pool_c = self._get_C(max_pool)
      if max_pool_c != out_filters:
        with fw.variable_scope("conv"):
          w = fw.create_weight(
            "w", [num_possible_inputs, max_pool_c * out_filters])
          w = w[prev_cell]
          w = fw.reshape(w, [1, 1, max_pool_c, out_filters])
          max_pool = fw.relu(max_pool)
          max_pool = fw.conv2d(max_pool, w, strides=[1, 1, 1, 1],
                                  padding="SAME", data_format=self.data_format)
          max_pool = batch_norm(max_pool, is_training=True,
                                data_format=self.data_format)

    x_c = self._get_C(x)
    if x_c != out_filters:
      with fw.variable_scope("x_conv"):
        w = fw.create_weight("w", [num_possible_inputs, x_c * out_filters])
        w = w[prev_cell]
        w = fw.reshape(w, [1, 1, x_c, out_filters])
        x = fw.relu(x)
        x = fw.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME",
                         data_format=self.data_format)
        x = batch_norm(x, is_training=True, data_format=self.data_format)

    out = [
      self._enas_conv(x, curr_cell, prev_cell, 3, out_filters),
      self._enas_conv(x, curr_cell, prev_cell, 5, out_filters),
      avg_pool,
      max_pool,
      x,
    ]

    out = fw.stack(out, axis=0)
    out = out[op_id, :, :, :, :]
    return out

  def _enas_conv(self, x, curr_cell, prev_cell, filter_size, out_filters,
                 stack_conv=2):
    """Performs an enas convolution specified by the relevant parameters."""

    with fw.variable_scope("conv_{0}x{0}".format(filter_size)):
      num_possible_inputs = curr_cell + 2
      for conv_id in range(stack_conv):
        with fw.variable_scope("stack_{0}".format(conv_id)):
          # create params and pick the correct path
          inp_c = self._get_C(x)
          w_depthwise = fw.create_weight(
            "w_depth", [num_possible_inputs, filter_size * filter_size * inp_c])
          w_depthwise = w_depthwise[prev_cell, :]
          w_depthwise = fw.reshape(
            w_depthwise, [filter_size, filter_size, inp_c, 1])

          w_pointwise = fw.create_weight(
            "w_point", [num_possible_inputs, inp_c * out_filters])
          w_pointwise = w_pointwise[prev_cell, :]
          w_pointwise = fw.reshape(w_pointwise, [1, 1, inp_c, out_filters])

          with fw.variable_scope("bn"):
            zero_init = fw.zeros_init()
            one_init = fw.ones_init()
            offset = fw.create_weight(
              "offset", [num_possible_inputs, out_filters],
              initializer=zero_init)
            scale = fw.create_weight(
              "scale", [num_possible_inputs, out_filters],
              initializer=one_init)
            offset = offset[prev_cell]
            scale = scale[prev_cell]

          # the computations
          x = fw.relu(x)
          x = fw.separable_conv2d(
            x,
            depthwise_filter=w_depthwise,
            pointwise_filter=w_pointwise,
            strides=[1, 1, 1, 1], padding="SAME",
            data_format=self.data_format)
          x, _, _ = fw.fused_batch_norm(
            x, scale, offset, epsilon=1e-5, data_format=self.data_format,
            is_training=True)
    return x

  def _enas_layer(self, layer_id, prev_layers, arc, out_filters):
    """
    Args:
      layer_id: current layer
      prev_layers: cache of previous layers. for skip connections
      start_idx: where to start looking at. technically, we can infer this
        from layer_id, but why bother...
    """

    assert len(prev_layers) == 2, "need exactly 2 inputs"
    layers = [prev_layers[0], prev_layers[1]]
    layers = self._maybe_calibrate_size(layers, out_filters, is_training=True)
    used = []
    for cell_id in range(self.num_cells):
      prev_layers = fw.stack(layers, axis=0)
      with fw.variable_scope("cell_{0}".format(cell_id)):
        with fw.variable_scope("x"):
          x_id = arc[4 * cell_id]
          x_op = arc[4 * cell_id + 1]
          x = prev_layers[x_id, :, :, :, :]
          x = self._enas_cell(x, cell_id, x_id, x_op, out_filters)
          x_used = fw.one_hot(x_id, depth=self.num_cells + 2, dtype=fw.int32)

        with fw.variable_scope("y"):
          y_id = arc[4 * cell_id + 2]
          y_op = arc[4 * cell_id + 3]
          y = prev_layers[y_id, :, :, :, :]
          y = self._enas_cell(y, cell_id, y_id, y_op, out_filters)
          y_used = fw.one_hot(y_id, depth=self.num_cells + 2, dtype=fw.int32)

        out = x + y
        used.extend([x_used, y_used])
        layers.append(out)

    used = fw.add_n(used)
    indices = fw.where(fw.equal(used, 0))
    indices = fw.to_int32(indices)
    indices = fw.reshape(indices, [-1])
    num_outs = fw.size(indices)
    out = fw.stack(layers, axis=0)
    out = fw.gather(out, indices, axis=0)

    inp = prev_layers[0]
    if self.data_format == "NHWC":
      N = fw.shape(inp)[0]
      H = fw.shape(inp)[1]
      W = fw.shape(inp)[2]
      C = fw.shape(inp)[3]
      out = fw.transpose(out, [1, 2, 3, 0, 4])
      out = fw.reshape(out, [N, H, W, num_outs * out_filters])
    elif self.data_format == "NCHW":
      N = fw.shape(inp)[0]
      C = fw.shape(inp)[1]
      H = fw.shape(inp)[2]
      W = fw.shape(inp)[3]
      out = fw.transpose(out, [1, 0, 2, 3, 4])
      out = fw.reshape(out, [N, num_outs * out_filters, H, W])
    else:
      raise ValueError("Unknown data_format '{0}'".format(self.data_format))

    with fw.variable_scope("final_conv"):
      w = fw.create_weight("w", [self.num_cells + 2, out_filters * out_filters])
      w = fw.gather(w, indices, axis=0)
      w = fw.reshape(w, [1, 1, num_outs * out_filters, out_filters])
      out = fw.relu(out)
      out = fw.conv2d(out, w, strides=[1, 1, 1, 1], padding="SAME",
                         data_format=self.data_format)
      out = batch_norm(out, is_training=True, data_format=self.data_format)

    out = fw.reshape(out, fw.shape(prev_layers[0]))

    return out

  # override
  def _build_train(self):
    print("-" * 80)
    print("Build train graph")
    logits = self._model(self.x_train, is_training=True)
    log_probs = fw.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=self.y_train)
    self.loss = fw.reduce_mean(log_probs)

    if self.use_aux_heads:
      log_probs = fw.sparse_softmax_cross_entropy_with_logits(
        logits=self.aux_logits, labels=self.y_train)
      self.aux_loss = fw.reduce_mean(log_probs)
      train_loss = self.loss + 0.4 * self.aux_loss
    else:
      train_loss = self.loss

    self.train_preds = fw.argmax(logits, axis=1)
    self.train_preds = fw.to_int32(self.train_preds)
    self.train_acc = fw.equal(self.train_preds, self.y_train)
    self.train_acc = fw.to_int32(self.train_acc)
    self.train_acc = fw.reduce_sum(self.train_acc)

    tf_variables = [
      var for var in fw.trainable_variables() if (
        var.name.startswith(self.name) and "aux_head" not in var.name)]
    self.num_vars = count_model_params(tf_variables)
    print("Model has {0} params".format(self.num_vars))

    self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
      train_loss,
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
        self.seed,
        25000)

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

    logits = self._model(x_valid_shuffle, is_training=True, reuse=True)
    valid_shuffle_preds = fw.argmax(logits, axis=1)
    valid_shuffle_preds = fw.to_int32(valid_shuffle_preds)
    self.valid_shuffle_acc = fw.equal(valid_shuffle_preds, y_valid_shuffle)
    self.valid_shuffle_acc = fw.to_int32(self.valid_shuffle_acc)
    self.valid_shuffle_acc = fw.reduce_sum(self.valid_shuffle_acc)

  def connect_controller(self, controller_model):
    if self.fixed_arc is None:
      self.normal_arc, self.reduce_arc = controller_model.sample_arc
    else:
      fixed_arc = np.array([int(x) for x in self.fixed_arc.split(" ") if x])
      self.normal_arc = fixed_arc[:4 * self.num_cells]
      self.reduce_arc = fixed_arc[4 * self.num_cells:]

    self._build_train()
    self._build_valid()
    self._build_test()
