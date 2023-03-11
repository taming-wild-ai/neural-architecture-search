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

from src.utils import count_model_params
from src.utils import get_train_ops
from src.utils import DEFINE_boolean, DEFINE_float, DEFINE_integer, LayeredModel

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


  class SkipPath(LayeredModel):
    def __init__(self, stride_spec, data_format, weights, reuse: bool, scope: str, out_filters):
      self.layers = [
        lambda x: fw.avg_pool(
          x,
          [1, 1, 1, 1],
          stride_spec,
          "VALID",
          data_format=data_format.name),
        lambda x: fw.conv2d(
          x,
          weights.get(
            reuse,
            scope,
            "w",
            [1, 1, data_format.get_C(x), out_filters // 2],
            None),
          [1, 1, 1, 1],
          "VALID",
          data_format=data_format.name)]


  def _factorized_reduction(self, x, out_filters, stride, is_training, weights, reuse):
    """Reduces the shape of x without information loss due to striding."""
    assert out_filters % 2 == 0, (
        "Need even number of filters when using this factorized reduction.")
    if stride == 1:
      with fw.name_scope("path_conv") as scope:
        fr_model = Child.PathConv(
          weights,
          reuse,
          scope,
          out_filters,
          is_training,
          self.data_format)
        return fr_model(x)

    stride_spec = self.data_format.get_strides(stride)
    # Skip path 1
    with fw.name_scope("path1_conv") as scope:
      skip_path = MicroChild.SkipPath(stride_spec, self.data_format, weights, reuse, scope, out_filters)
      path1 = skip_path(x)

    # Skip path 2
    # First pad with 0"s on the right and bottom, then shift the filter to
    # include those 0"s that were added.
    path2, concat_axis = self.data_format.factorized_reduction(x)

    with fw.name_scope("path2_conv") as scope:
      skip_path = MicroChild.SkipPath(stride_spec, self.data_format, weights, reuse, scope, out_filters)
      path2 = skip_path(path2)

    # Concat and apply BN
    fr_model = Child.FactorizedReduction(concat_axis, is_training, self.data_format, weights)
    return fr_model([path1, path2])

  def _get_HW(self, x):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    return x.get_shape()[2]

  def _apply_drop_path(self, x, layer_id):
    return drop_path(
      x,
      1.0 - fw.minimum(
        1.0,
        fw.to_float(self.global_step + 1) / fw.to_float(self.num_train_steps)) * (1.0 - (1.0 - float(layer_id + 1) / (self.num_layers + 2) * (1.0 - self.drop_path_keep_prob))))

  def _maybe_calibrate_size(self, layers, out_filters, is_training, weights, reuse):
    """Makes sure layers[0] and layers[1] have the same shapes."""

    hw = [self._get_HW(layer) for layer in layers]
    c = [self.data_format.get_C(layer) for layer in layers]

    with fw.name_scope("calibrate") as scope:
      x = layers[0]
      if hw[0] != hw[1]:
        assert hw[0] == 2 * hw[1], f"hw[0] = {hw[0]}, hw[1] = {hw[1]}"
        with fw.name_scope("pool_x") as scope:
          x = fw.relu(x)
          x = self._factorized_reduction(x, out_filters, 2, is_training, weights, reuse)
      elif c[0] != out_filters:
        with fw.name_scope("pool_x") as scope:
          conv = Child.Conv1x1(weights, reuse, scope, c[0], out_filters, is_training, self.data_format)
          x = conv(x)

      y = layers[1]
      if c[1] != out_filters:
        with fw.name_scope("pool_y") as scope:
          y = batch_norm(
            fw.conv2d(
              fw.relu(y),
              weights.get(reuse, scope, "w", [1, 1, c[1], out_filters], None),
              [1, 1, 1, 1],
              "SAME",
              data_format=self.data_format.name),
            is_training,
            self.data_format,
            weights)
    return [x, y]


  class Dropout(LayeredModel):
    def __init__(self, data_format, is_training, keep_prob, weights, reuse, scope):
      self.layers = [
        fw.relu,
        data_format.global_avg_pool]
      if is_training and keep_prob is not None and keep_prob < 1.0:
        self.layers += [lambda x: fw.dropout(x, keep_prob)]
      self.layers += [
        lambda x: fw.matmul(
          x,
          weights.get(
            reuse,
            scope,
            "w",
            [data_format.get_C(x), 10],
            None))]


  class FullyConnected(LayeredModel):
    def __init__(self, data_format, weights, reuse, scope):
      self.layers = [
        data_format.global_avg_pool,
        lambda x: fw.matmul(
          x,
          weights.get(reuse, scope, "w", [x.get_shape()[1], 10], None))]


  def _model(self, images, is_training, reuse=False):
    """Compute the logits given the images."""

    if self.fixed_arc is None:
      is_training = True

    with fw.name_scope(self.name) as scope:
      # the first two inputs
      with fw.name_scope("stem_conv") as scope:
        stem_conv = Child.StemConv(self.weights, reuse, scope, self.out_filters * 3, is_training, self.data_format)
        x = stem_conv(images)
      layers = [x, x]

      # building layers in the micro space
      out_filters = self.out_filters
      for layer_id in range(self.num_layers + 2):
        with fw.name_scope("layer_{0}".format(layer_id)) as scope:
          if layer_id not in self.pool_layers:
            if self.fixed_arc is None:
              x = self._enas_layer(
                layer_id, layers, self.normal_arc, out_filters, self.weights, reuse)
            else:
              x = self._fixed_layer(
                layer_id, layers, self.normal_arc, out_filters, 1, is_training,
                self.weights, reuse, normal_or_reduction_cell="normal")
          else:
            out_filters *= 2
            if self.fixed_arc is None:
              x = self._factorized_reduction(x, out_filters, 2, is_training, self.weights, reuse)
              layers = [layers[-1], x]
              x = self._enas_layer(
                layer_id, layers, self.reduce_arc, out_filters, self.weights, reuse)
            else:
              x = self._fixed_layer(
                layer_id, layers, self.reduce_arc, out_filters, 2, is_training,
                self.weights, reuse, normal_or_reduction_cell="reduction")
          print("Layer {0:>2d}: {1}".format(layer_id, x))
          layers = [layers[-1], x]

        # auxiliary heads
        self.num_aux_vars = 0
        if (self.use_aux_heads and
            layer_id in self.aux_head_indices
            and is_training):
          print("Using aux_head at layer {0}".format(layer_id))
          with fw.name_scope("aux_head") as scope:
            aux_logits = fw.relu(x)
            aux_logits = fw.avg_pool2d(
              aux_logits, [5, 5], [3, 3], "VALID",
              data_format=self.data_format.actual)
            with fw.name_scope("proj") as scope:
              inp_conv = Child.InputConv(
                self.weights,
                reuse,
                scope,
                1,
                128,
                is_training,
                self.data_format)
              aux_logits = inp_conv(aux_logits)

            with fw.name_scope("avg_pool") as scope:
              hw = self._get_HW(aux_logits)
              inp_conv = Child.InputConv(
                self.weights,
                reuse,
                scope,
                hw,
                768,
                True,
                self.data_format)
              aux_logits = inp_conv(aux_logits)

            with fw.name_scope("fc") as scope:
              fc = MicroChild.FullyConnected(self.data_format, self.weights, reuse, scope)
              self.aux_logits = fc(aux_logits)

          aux_head_variables = [
            var for var in fw.trainable_variables() if (
              var.name.startswith(self.name) and "aux_head" in var.name)]
          self.num_aux_vars = count_model_params(aux_head_variables)
          print("Aux head uses {0} params".format(self.num_aux_vars))

      with fw.name_scope("fc") as scope:
        dropout = MicroChild.Dropout(
          self.data_format,
          is_training,
          self.keep_prob,
          self.weights,
          reuse,
          scope)
        x = dropout(x)
    return x


  class SeparableConv(LayeredModel):
    def __init__(self, weights, reuse, scope, filter_size, inp_c, data_format, out_filters, strides, is_training):
      self.layers = [
        fw.relu,
        lambda x: fw.separable_conv2d(
          x,
          depthwise_filter=weights.get(reuse, scope, "w_depth", [filter_size, filter_size, inp_c, 1], None),
          pointwise_filter=weights.get(reuse, scope, "w_point", [1, 1, inp_c, out_filters], None),
          strides=strides,
          padding="SAME",
          data_format=data_format.name),
        lambda x: batch_norm(x, is_training, data_format, weights)]


  def _fixed_conv(self, x, f_size, out_filters, stride, is_training, weights, reuse,
                  stack_convs=2):
    """Apply fixed convolution.

    Args:
      stacked_convs: number of separable convs to apply.
    """

    for conv_id in range(stack_convs):
      inp_c = self.data_format.get_C(x)
      if conv_id == 0:
        strides = self.data_format.get_strides(stride)
      else:
        strides = [1, 1, 1, 1]

      with fw.name_scope("sep_conv_{}".format(conv_id)) as scope:
        sep_conv = MicroChild.SeparableConv(
          weights,
          reuse,
          scope,
          f_size,
          inp_c,
          self.data_format,
          out_filters,
          strides,
          is_training)
        x = sep_conv(x)

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

    with fw.name_scope("final_combine") as scope:
      for i, layer in enumerate(layers):
        if used[i] == 0:
          hw = self._get_HW(layer)
          if hw > out_hw:
            assert hw == out_hw * 2, ("i_hw={0} != {1}=o_hw".format(hw, out_hw))
            with fw.name_scope("calibrate_{0}".format(i)) as scope:
              x = self._factorized_reduction(layer, out_filters, 2, is_training)
          else:
            x = layer
          out.append(x)
      out = self.data_format.fixed_layer(out)

    return out


  class LayerBase(LayeredModel):
    def __init__(self, weights, reuse: bool, scope: str, out_filters: int, is_training: bool, data_format):
      self.layers = [
        fw.relu,
        lambda x: fw.conv2d(
          x,
          weights.get(reuse, scope, "w", [1, 1, data_format.get_C(x), out_filters], None),
          [1, 1, 1, 1],
          "SAME",
          data_format=data_format.name),
        lambda x: batch_norm(x, is_training, data_format, weights)]

  def _fixed_layer(self, layer_id, prev_layers, arc, out_filters, stride,
                   is_training, weights, reuse, normal_or_reduction_cell="normal"):
    """
    Args:
      prev_layers: cache of previous layers. for skip connections
      is_training: for batch_norm
    """

    assert len(prev_layers) == 2
    layers = [prev_layers[0], prev_layers[1]]
    layers = self._maybe_calibrate_size(layers, out_filters,
                                        is_training, weights, reuse)

    with fw.name_scope("layer_base") as scope:
      lb = MicroChild.LayerBase(weights, reuse, scope, out_filters, is_training, self.data_format)
      layers[1] = lb(layers[1])

    used = np.zeros([self.num_cells + 2], dtype=np.int32)
    f_sizes = [3, 5]
    for cell_id in range(self.num_cells):
      with fw.name_scope("cell_{}".format(cell_id)) as scope:
        x_id = arc[4 * cell_id]
        used[x_id] += 1
        x_op = arc[4 * cell_id + 1]
        x = layers[x_id]
        x_stride = stride if x_id in [0, 1] else 1
        with fw.name_scope("x_conv") as scope:
          if x_op in [0, 1]:
            f_size = f_sizes[x_op]
            x = self._fixed_conv(x, f_size, out_filters, x_stride, is_training, weights, reuse)
          elif x_op in [2, 3]:
            inp_c = self.data_format.get_C(x)
            if x_op == 2:
              x = fw.avg_pool2d(
                x, [3, 3], [x_stride, x_stride], "SAME",
                data_format=self.data_format.actual)
            else:
              x = fw.max_pool2d(
                x, [3, 3], [x_stride, x_stride], "SAME",
                data_format=self.data_format.actual)
            if inp_c != out_filters:
              x_conv = Child.Conv1x1(weights, reuse, scope, inp_c, out_filters, is_training, self.data_format)
              x = x_conv(x)
          else:
            inp_c = self.data_format.get_C(x)
            if x_stride > 1:
              assert x_stride == 2
              x = self._factorized_reduction(x, out_filters, 2, is_training)
            if inp_c != out_filters:
              x_conv = Child.Conv1x1(weights, reuse, scope, inp_c, out_filters, is_training, self.data_format)
              x = x_conv(x)
          if (x_op in [0, 1, 2, 3] and
              self.drop_path_keep_prob is not None and
              is_training):
            x = self._apply_drop_path(x, layer_id)

        y_id = arc[4 * cell_id + 2]
        used[y_id] += 1
        y_op = arc[4 * cell_id + 3]
        y = layers[y_id]
        y_stride = stride if y_id in [0, 1] else 1
        with fw.name_scope("y_conv") as scope:
          if y_op in [0, 1]:
            f_size = f_sizes[y_op]
            y = self._fixed_conv(y, f_size, out_filters, y_stride, is_training, weights, reuse)
          elif y_op in [2, 3]:
            inp_c = self.data_format.get_C(y)
            if y_op == 2:
              y = fw.avg_pool2d(
                y, [3, 3], [y_stride, y_stride], "SAME",
                data_format=self.data_format.actual)
            else:
              y = fw.max_pool2d(
                y, [3, 3], [y_stride, y_stride], "SAME",
                data_format=self.data_format.actual)
            if inp_c != out_filters:
              out_conv = Child.Conv1x1(weights, reuse, scope, inp_c, out_filters, is_training, self.data_format)
              y = out_conv(y)
          else:
            inp_c = self.data_format.get_C(y)
            if y_stride > 1:
              assert y_stride == 2
              y = self._factorized_reduction(y, out_filters, 2, is_training, weights, reuse)
            if inp_c != out_filters:
              y = batch_norm(
                fw.conv2d(
                  fw.relu(y),
                  weights.get(reuse, scope, "w", [1, 1, inp_c, out_filters], None),
                  [1, 1, 1, 1],
                  "SAME",
                  data_format=self.data_format.name),
                is_training,
                self.data_format,
                weights)

          if (y_op in [0, 1, 2, 3] and
              self.drop_path_keep_prob is not None and
              is_training):
            y = self._apply_drop_path(y, layer_id)

        out = x + y
        layers.append(out)
    out = self._fixed_combine(layers, used, out_filters, is_training,
                              normal_or_reduction_cell)

    return out


  class MaxPool(LayeredModel):
    def __init__(self, data_format, out_filters: int, reuse: bool, scope: str, curr_cell, prev_cell: int, weights):
      def inner(x):
        inp_c = data_format.get_C(x)
        if inp_c != out_filters:
          with fw.name_scope("conv"):
            return batch_norm(
              fw.conv2d(
                fw.relu(x),
                fw.reshape(
                  weights.get(
                    reuse,
                    scope,
                    'w',
                    [curr_cell + 1, inp_c * out_filters],
                    None)[prev_cell],
                  [1, 1, inp_c, out_filters]),
                strides=[1, 1, 1, 1],
                padding="SAME",
                data_format=data_format.name),
              True,
              data_format,
              weights)
        else:
          return x
      self.layers = [
        lambda x: fw.max_pool2d(x, [3, 3], [1, 1], 'SAME', data_format=data_format.actual),
        inner]


  class AvgPool(LayeredModel):
    def __init__(self, data_format, out_filters: int, reuse: bool, scope: str, curr_cell, prev_cell: int, weights):
      def inner(x):
        inp_c = data_format.get_C(x)
        if inp_c != out_filters:
          with fw.name_scope("conv"):
            return batch_norm(
              fw.conv2d(
                fw.relu(x),
                fw.reshape(
                  weights.get(
                    reuse,
                    scope,
                    'w',
                    [curr_cell + 1, inp_c * out_filters],
                    None)[prev_cell],
                  [1, 1, inp_c, out_filters]),
                strides=[1, 1, 1, 1],
                padding="SAME",
                data_format=data_format.name),
              True,
              data_format,
              weights)
        else:
          return x
      self.layers = [
        lambda x: fw.avg_pool2d(x, [3, 3], [1, 1], 'SAME', data_format=data_format.actual),
        inner]

  def _enas_cell(self, x, curr_cell, prev_cell, op_id, out_filters, weights, reuse: bool):
    """Performs an enas operation specified by op_id."""

    num_possible_inputs = curr_cell + 1

    with fw.name_scope("avg_pool") as scope:
      ap = MicroChild.AvgPool(self.data_format, out_filters, reuse, scope, curr_cell, prev_cell, weights)
      avg_pool = ap(x)

    with fw.name_scope("max_pool") as scope:
      mp = MicroChild.MaxPool(self.data_format, out_filters, reuse, scope, curr_cell, prev_cell, weights)
      max_pool = mp(x)

    x_c = self.data_format.get_C(x)
    if x_c != out_filters:
      with fw.name_scope("x_conv") as scope:
        x = batch_norm(
          fw.conv2d(
            fw.relu(x),
            fw.reshape(
              weights.get(
                reuse,
                scope,
                "w",
                [num_possible_inputs, x_c * out_filters],
                None)[prev_cell],
              [1, 1, x_c, out_filters]),
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format=self.data_format.name),
          True,
          self.data_format,
          weights)

    out = [
      self._enas_conv(x, curr_cell, prev_cell, 3, out_filters, weights, reuse),
      self._enas_conv(x, curr_cell, prev_cell, 5, out_filters, weights, reuse),
      avg_pool,
      max_pool,
      x,
    ]

    out = fw.stack(out, axis=0)
    out = out[op_id, :, :, :, :]
    return out

  def _enas_conv(self, x, curr_cell, prev_cell, filter_size, out_filters, weights, reuse: bool,
                 stack_conv=2):
    """Performs an enas convolution specified by the relevant parameters."""

    with fw.name_scope("conv_{0}x{0}".format(filter_size)) as scope:
      num_possible_inputs = curr_cell + 2
      for conv_id in range(stack_conv):
        with fw.name_scope("stack_{0}".format(conv_id)) as scope:
          # create params and pick the correct path
          inp_c = self.data_format.get_C(x)
          w_depthwise = fw.reshape(
            weights.get(
              reuse,
              scope,
              "w_depth",
              [num_possible_inputs, filter_size * filter_size * inp_c], None)[prev_cell, :],
            [filter_size, filter_size, inp_c, 1])

          w_pointwise = fw.reshape(
            weights.get(
              reuse,
              scope,
              "w_point",
              [num_possible_inputs, inp_c * out_filters], None)[prev_cell, :],
            [1, 1, inp_c, out_filters])

          with fw.name_scope("bn") as scope:
            offset = weights.get(
              reuse,
              scope,
              "offset", [num_possible_inputs, out_filters],
              fw.zeros_init())[prev_cell]
            scale = weights.get(
              reuse,
              scope,
              "scale", [num_possible_inputs, out_filters],
              fw.ones_init())[prev_cell]

          # the computations
          x, _, _ = fw.fused_batch_norm(
            fw.separable_conv2d(
              fw.relu(x),
              depthwise_filter=w_depthwise,
              pointwise_filter=w_pointwise,
              strides=[1, 1, 1, 1],
              padding="SAME",
              data_format=self.data_format.name),
            scale,
            offset,
            epsilon=1e-5,
            data_format=self.data_format.name,
            is_training=True)
    return x

  def _enas_layer(self, layer_id, prev_layers, arc, out_filters, weights, reuse):
    """
    Args:
      layer_id: current layer
      prev_layers: cache of previous layers. for skip connections
      start_idx: where to start looking at. technically, we can infer this
        from layer_id, but why bother...
    """

    assert len(prev_layers) == 2, "need exactly 2 inputs"
    layers = [prev_layers[0], prev_layers[1]]
    layers = self._maybe_calibrate_size(layers, out_filters, True, weights, reuse)
    used = []
    for cell_id in range(self.num_cells):
      prev_layers = fw.stack(layers, axis=0)
      with fw.name_scope("cell_{0}".format(cell_id)) as scope:
        with fw.name_scope("x") as scope:
          x_id = arc[4 * cell_id]
          x_op = arc[4 * cell_id + 1]
          x = prev_layers[x_id, :, :, :, :]
          x = self._enas_cell(x, cell_id, x_id, x_op, out_filters, weights, reuse)
          x_used = fw.one_hot(x_id, depth=self.num_cells + 2, dtype=fw.int32)

        with fw.name_scope("y") as scope:
          y_id = arc[4 * cell_id + 2]
          y_op = arc[4 * cell_id + 3]
          y = prev_layers[y_id, :, :, :, :]
          y = self._enas_cell(y, cell_id, y_id, y_op, out_filters, weights, reuse)
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
    out = self.data_format.micro_enas(out, inp, num_outs, out_filters)

    with fw.name_scope("final_conv") as scope:
      out = batch_norm(
        fw.conv2d(
          fw.relu(out),
          fw.reshape(
            fw.gather(
              weights.get(
                reuse,
                scope,
                "w",
                [self.num_cells + 2, out_filters * out_filters],
                None),
              indices,
              axis=0),
            [1, 1, num_outs * out_filters, out_filters]),
          strides=[1, 1, 1, 1],
          padding="SAME",
          data_format=self.data_format.name),
        True,
        self.data_format,
        weights)

    out = fw.reshape(out, fw.shape(prev_layers[0]))

    return out

  # override
  def _build_train(self):
    print("-" * 80)
    print("Build train graph")
    logits = self._model(self.x_train, is_training=True)
    self.loss = fw.reduce_mean(fw.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=self.y_train))

    if self.use_aux_heads:
      self.aux_loss = fw.reduce_mean(fw.sparse_softmax_cross_entropy_with_logits(
        logits=self.aux_logits, labels=self.y_train))
      train_loss = self.loss + 0.4 * self.aux_loss
    else:
      train_loss = self.loss

    self.train_acc = fw.reduce_sum(fw.to_int32(fw.equal(fw.to_int32(fw.argmax(logits, axis=1)), self.y_train)))

    tf_variables = [
      var for var in fw.trainable_variables() if (
        var.name.startswith(self.name) and "aux_head" not in var.name)]
    self.num_vars = count_model_params(tf_variables)
    print("Model has {0} params".format(self.num_vars))

    self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
      train_loss,
      tf_variables,
      self.global_step,
      self.learning_rate,
      clip_mode=self.clip_mode,
      l2_reg=self.l2_reg,
      num_train_batches=self.num_train_batches,
      optim_algo=self.optim_algo)

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
      if not shuffle:
        self.images["valid_original"] = self.data_format.child_init(self.images['valid_original'])
      x_valid_shuffle, y_valid_shuffle = fw.shuffle_batch(
        [self.images["valid_original"], self.labels["valid_original"]],
        self.batch_size,
        self.seed,
        25000)

      def _pre_process(x):
        return self.data_format.child_init_preprocess(
          fw.image.random_flip_left_right(
            fw.image.random_crop(
              fw.pad(x, [[4, 4], [4, 4], [0, 0]]),
              [32, 32, 3],
              seed=self.seed),
            seed=self.seed))

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
