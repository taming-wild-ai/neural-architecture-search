from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
from absl import flags
import src.framework as fw

from src.cifar10.child import Child
from src.cifar10.image_ops import BatchNorm
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
    FLAGS = flags.FLAGS

    self.use_aux_heads = FLAGS.child_use_aux_heads
    self.num_epochs = FLAGS.num_epochs
    self.num_train_steps = self.num_epochs * self.num_train_batches
    self.drop_path_keep_prob = FLAGS.child_drop_path_keep_prob
    self.lr_min = lr_min
    self.num_cells = FLAGS.child_num_cells

    if self.drop_path_keep_prob is not None:
      assert self.num_epochs is not None, "Need num_epochs to drop_path"

    pool_distance = self.num_layers // 3
    self.pool_layers = [pool_distance, 2 * pool_distance + 1]

    if self.use_aux_heads:
      self.aux_head_indices = [self.pool_layers[-1] + 1]


  class SkipPath(LayeredModel):
    def __init__(self, stride_spec, data_format, weights, reuse: bool, scope: str, num_input_chan: int, out_filters):
      w = weights.get(
        reuse,
        scope,
        "w",
        [1, 1, num_input_chan, out_filters // 2],
        None)
      def avg_pool(x):
        return fw.avg_pool(
          x,
          [1, 1, 1, 1],
          stride_spec,
          "VALID",
          data_format=data_format.name)
      def conv2d(x):
        return fw.conv2d(
          x,
          w,
          [1, 1, 1, 1],
          "VALID", # Only difference from MacroChild.SkipPath
          data_format=data_format.name)
      self.layers = [avg_pool, conv2d]


  def _apply_drop_path(self, x, layer_id):
    return drop_path(
      x,
      1.0 - fw.minimum(
        1.0,
        fw.to_float(self.global_step + 1) / fw.to_float(self.num_train_steps)) * (1.0 - (1.0 - float(layer_id + 1) / (self.num_layers + 2) * (1.0 - self.drop_path_keep_prob))))


  # Because __call__ is overridden, this superclass is just for ease of find.
  class CalibrateSize(LayeredModel):
    def __init__(self, child, hw, c, out_filters, is_training: bool, weights, reuse: bool):
      with fw.name_scope("calibrate") as scope:
        if hw[0] != hw[1]:
          assert hw[0] == 2 * hw[1], f"hw[0] = {hw[0]}, hw[1] = {hw[1]}"
          with fw.name_scope("pool_x") as scope:
            self.layers_x = [
              fw.relu,
              Child.FactorizedReduction(child, c[0], out_filters, 2, is_training, weights, reuse)]
        elif c[0] != out_filters:
          with fw.name_scope("pool_x") as scope:
            self.layers_x = [
              Child.Conv1x1(
                weights,
                reuse,
                scope,
                c[0],
                out_filters,
                is_training,
                child.data_format)]
        else:
          self.layers_x = []
        if c[1] != out_filters:
          with fw.name_scope("pool_y") as scope:
            w = weights.get(reuse, scope, "w", [1, 1, c[1], out_filters], None)
            bn = BatchNorm(is_training, child.data_format, weights, out_filters, reuse)
            self.layers_y = [
              fw.relu,
              lambda x: fw.conv2d(
                x,
                w,
                [1, 1, 1, 1],
                "SAME",
                data_format=child.data_format.name),
              bn]
        else:
          self.layers_y = []

    def __call__(self, layers):
      x = layers[0]
      y = layers[1]

      with fw.name_scope("calibrate"):
        with fw.name_scope("pool_x"):
          for layer in self.layers_x:
            x = layer(x)
        with fw.name_scope("pool_y"):
          for layer in self.layers_y:
            y = layer(y)

      return [x, y]


  class Dropout(LayeredModel):
    def __init__(self, data_format, is_training, keep_prob, weights, reuse, scope: str, num_input_chan: int):
      w = weights.get(
        reuse,
        scope,
        "w",
        [num_input_chan, 10],
        None)
      def matmul(x):
        return fw.matmul(
          x,
          w)
      self.layers = [
        fw.relu,
        data_format.global_avg_pool]
      if is_training and keep_prob is not None and keep_prob < 1.0:
        self.layers += [lambda x: fw.dropout(x, keep_prob)]
      self.layers += [matmul]


  class FullyConnected(LayeredModel):
    def __init__(self, data_format, weights, reuse, scope, num_input_chan):
      w = weights.get(reuse, scope, "w", [num_input_chan, 10], None)
      self.layers = [
        data_format.global_avg_pool,
        lambda x: fw.matmul(x, w)]


  # Because __call__ is overridden, this superclass is just for ease of find.
  class Model(LayeredModel):
    def __init__(self, child, is_training, reuse=False):
      self.child = child
      self.trainable_variables = child.trainable_variables
      self.layers = {}
      self.aux_logits = {}
      if child.fixed_arc is None:
        is_training = True
      with fw.name_scope(child.name) as scope:
        with fw.name_scope('stem_conv') as scope:
          self.stem_conv = Child.StemConv(child.weights, reuse, scope, child.out_filters * 3, is_training, child.data_format)
        x_chan = child.out_filters * 3
        layers_channels = [x_chan, x_chan]
        layers_hw = [32, 32]
        out_filters = child.out_filters
        for layer_id in range(child.num_layers + 2):
          with fw.name_scope(f'layer_{layer_id}') as scope:
            if layer_id not in child.pool_layers:
              if child.fixed_arc is None:
                self.layers[layer_id] = MicroChild.ENASLayer(child, child.current_controller_normal_arc(), layers_hw, layers_channels, out_filters, child.weights, reuse)
                x_chan = out_filters
              else:
                self.layers[layer_id] = MicroChild.FixedLayer(child, layer_id, child.current_controller_normal_arc(), layers_hw, layers_channels, out_filters, 1, is_training, child.weights, reuse)
                x_chan = self.layers[layer_id].out_chan
              x_hw = layers_hw[-1]
            else:
              out_filters *= 2
              if child.fixed_arc is None:
                self.layers[layer_id] = [Child.FactorizedReduction(child, x_chan, out_filters, 2, is_training, child.weights, reuse)]
                x_hw = layers_hw[-1] // 2
                layers_hw = [layers_hw[-1], x_hw]
                layers_channels = [layers_channels[-1], out_filters]
                self.layers[layer_id].append(MicroChild.ENASLayer(child, child.current_controller_reduce_arc(), layers_hw, layers_channels, out_filters, child.weights, reuse))
                x_chan = out_filters
              else:
                self.layers[layer_id] = MicroChild.FixedLayer(child, layer_id, child.current_controller_reduce_arc(), layers_hw, layers_channels, out_filters, 2, is_training, child.weights, reuse)
                x_chan = self.layers[layer_id].out_chan
                x_hw = layers_hw[-1] // 2
            layers_hw = [layers_hw[-1], x_hw]
            layers_channels = [layers_channels[-1], x_chan]
          child.num_aux_vars = 0
          if (child.use_aux_heads and layer_id in child.aux_head_indices and is_training):
            print(f'Using aux_head at layer {layer_id}')
            with fw.name_scope('aux_head') as scope:
              self.aux_logits[layer_id] = [
                fw.relu,
                fw.avg_pool2d(
                  [5, 5],
                  [3, 3],
                  "VALID",
                  data_format=child.data_format.actual)]
              with fw.name_scope('proj') as scope:
                self.aux_logits[layer_id].append(Child.InputConv(
                  child.weights,
                  reuse,
                  scope,
                  1,
                  x_chan,
                  128,
                  is_training,
                  child.data_format))
              with fw.name_scope("avg_pool") as scope:
                self.aux_logits[layer_id].append(Child.InputConv(
                  child.weights,
                  reuse,
                  scope,
                  2, # self._get_HW(aux_logits)
                  128,
                  768,
                  True,
                  child.data_format))
              with fw.name_scope('fc') as scope:
                self.aux_logits[layer_id].append(MicroChild.FullyConnected(child.data_format, child.weights, reuse, scope, 768))
        aux_head_variables = [var for _, var in child.weights.weight_map.items() if var.trainable and var.name.startswith(child.name) and "aux_head" in var.name]
        num_aux_vars = count_model_params(aux_head_variables)
        print("Aux head uses {0} params".format(num_aux_vars))
        with fw.name_scope('fc') as scope:
          self.dropout = MicroChild.Dropout(
            child.data_format,
            is_training,
            child.keep_prob,
            child.weights,
            reuse,
            scope,
            x_chan)
        print("Model has {0} params".format(count_model_params(child.trainable_variables())))

    def __call__(self, images):
      """images should be a tf.data.Dataset batch."""
      with fw.name_scope(self.child.name):
        with fw.name_scope('stem_conv'):
          logits = self.stem_conv(images)
          layers = [logits, logits]
        aux_logits = None
        for layer_id in range(self.child.num_layers + 2):
          with fw.name_scope(f'layer_{layer_id}'):
            if layer_id not in self.child.pool_layers:
              logits = self.layers[layer_id](layers)
            else:
              if self.child.fixed_arc is None:
                logits = self.layers[layer_id][0](logits)
                layers = [layers[-1], logits]
                logits = self.layers[layer_id][1](layers)
              else:
                logits = self.layers[layer_id](layers)
            print("Layer {0:>2d}: {1}".format(layer_id, logits))
            layers = [layers[-1], logits]
          aux_logit_fns = self.aux_logits.get(layer_id)
          if aux_logit_fns:
            with fw.name_scope('aux_head'):
              aux_logits = aux_logit_fns[0](logits)
              with fw.name_scope('proj'):
                aux_logits = aux_logit_fns[1](aux_logits)
              with fw.name_scope('avg_pool'):
                aux_logits = aux_logit_fns[2](aux_logits)
              with fw.name_scope('fc'):
                aux_logits = aux_logit_fns[3](aux_logits)
        with fw.name_scope('fc'):
          logits = self.dropout(logits)
      return logits, aux_logits


  class SeparableConv(LayeredModel):
    def __init__(self, weights, reuse, scope, filter_size, data_format, num_input_chan: int, out_filters: int, strides, is_training):
      depthwise_filter=weights.get(reuse, scope, "w_depth", [filter_size, filter_size, num_input_chan, 1], None)
      pointwise_filter=weights.get(reuse, scope, "w_point", [1, 1, num_input_chan, out_filters], None)
      def separable_conv2d(x):
        return fw.separable_conv2d(
          fw.relu(x),
          depthwise_filter=depthwise_filter,
          pointwise_filter=pointwise_filter,
          strides=strides,
          padding="SAME",
          data_format=data_format.name)
      bn = BatchNorm(is_training, data_format, weights, out_filters, reuse)
      self.layers = [separable_conv2d, bn]


  class FixedConv(LayeredModel):
    """Apply fixed convolution.

    Args:
      stacked_convs: number of separable convs to apply.
    """
    def __init__(self, child, f_size, num_input_chan: int, out_filters: int, stride, is_training:bool, weights, reuse:bool, stack_convs=2):
      self.layers = []
      for conv_id in range(stack_convs):
        if 0 == conv_id:
          strides = child.data_format.get_strides(stride)
        else:
          strides = [1, 1, 1, 1]

        with fw.name_scope(f"sep_conv_{conv_id}") as scope:
          self.layers.append(
            MicroChild.SeparableConv(
              weights,
              reuse,
              scope,
              f_size,
              child.data_format,
              num_input_chan,
              out_filters,
              strides,
              is_training))


  # Because __call__ is overridden, this superclass is just for ease of find.
  class FixedCombine(LayeredModel):
    def __init__(self, child, used, c, in_hws, out_hw: int, out_filters: int, is_training, weights, reuse):
      self.layers = {}
      with fw.name_scope('final_combine'):
        for i in range(len(used)):
          if used[i] == 0:
            hw = in_hws[i]
            if hw > out_hw:
              assert hw == out_hw * 2, f"hw ({hw}) is not two times out_hw ({out_hw})"
              with fw.name_scope(f'calibrate_{i}'):
                self.layers[i] = Child.FactorizedReduction(child, c[i], out_filters, 2, is_training, weights, reuse)
            else:
              assert hw == out_hw, f"expected {out_hw}, got {hw}"
              self.layers[i] = lambda x: x

      def layer_fn(layers):
        out = []
        with fw.name_scope('final_combine'):
          for i, layer in enumerate(layers):
            if used[i] == 0:
              x = self.layers[i](layer)
              out.append(x)
          return out
      self.first_layer = layer_fn
      self.last_layer = child.data_format.fixed_layer

    def __call__(self, layers):
      out = self.first_layer(layers)
      with fw.name_scope('final_combine'):
        return self.last_layer(out)


  class LayerBase(LayeredModel):
    def __init__(self, weights, reuse: bool, scope: str, num_input_chan: int, out_filters: int, is_training: bool, data_format):
      w = weights.get(reuse, scope, "w", [1, 1, num_input_chan, out_filters], None)
      bn = BatchNorm(is_training, data_format, weights, out_filters, reuse)
      self.layers = [
        fw.relu,
        lambda x: fw.conv2d(
          x,
          w,
          [1, 1, 1, 1],
          "SAME",
          data_format=data_format.name),
        bn]


  class Operator(object):
    def inner2(x, child, is_training, layer_id):
      if (child.drop_path_keep_prob is not None and
          is_training):
        return child._apply_drop_path(x, layer_id)
      else:
        return x

    class SeparableConv3x3(LayeredModel):
      def __init__(self, child, num_input_chan, out_filters, x_stride, is_training: bool, weights, reuse: bool, scope: str, layer_id: int):
        self.layers = [
          MicroChild.FixedConv(child, 3, num_input_chan, out_filters, x_stride, is_training, weights, reuse),
          lambda x: MicroChild.Operator.inner2(x, child, is_training, layer_id)]


    class SeparableConv5x5(LayeredModel):
      def __init__(self, child, num_input_chan, out_filters, x_stride, is_training: bool, weights, reuse: bool, scope: str, layer_id: int):
        self.layers = [
          MicroChild.FixedConv(child, 5, num_input_chan, out_filters, x_stride, is_training, weights, reuse),
          lambda x: MicroChild.Operator.inner2(x, child, is_training, layer_id)]


    class AveragePooling(LayeredModel):
      def __init__(self, child, num_input_chan, out_filters, x_stride, is_training: bool, weights, reuse: bool, scope: str, layer_id: int):
        if num_input_chan != out_filters:
          x_conv = Child.Conv1x1(weights, reuse, scope, num_input_chan, out_filters, is_training, child.data_format)
          inner1 = lambda x: x_conv(x)
        else:
          inner1 = lambda x: x
        self.layers = [
          fw.avg_pool2d([3, 3], [x_stride, x_stride], "SAME", data_format=child.data_format.actual),
          inner1,
          lambda x: MicroChild.Operator.inner2(x, child, is_training, layer_id)]


    class MaxPooling(LayeredModel):
      def __init__(self, child, num_input_chan, out_filters, x_stride, is_training: bool, weights, reuse: bool, scope: str, layer_id: int):
        if num_input_chan != out_filters:
          x_conv = Child.Conv1x1(weights, reuse, scope, num_input_chan, out_filters, is_training, child.data_format)
          inner1 = lambda x: x_conv(x)
        else:
          inner1 = lambda x: x
        self.layers = [
          fw.max_pool2d([3, 3], [x_stride, x_stride], "SAME", data_format=child.data_format.actual),
          inner1,
          lambda x: MicroChild.Operator.inner2(x, child, is_training, layer_id)]


    class Identity(LayeredModel):
      def __init__(self, child, num_input_chan: int, out_filters: int, x_stride, is_training: bool, weights, reuse: bool, scope: str, layer_id: int):
        self.layers = []
        if x_stride > 1:
          assert x_stride == 2
          self.layers.append(Child.FactorizedReduction(child, num_input_chan, out_filters, 2, is_training, weights, reuse))
        if num_input_chan != out_filters:
          x_conv = Child.Conv1x1(weights, reuse, scope, num_input_chan, out_filters, is_training, child.data_format)
          inner1 = lambda x: x_conv(x)
        else:
          inner1 = lambda x: x
        self.layers.append(inner1)


    @staticmethod
    def new(op_id, child, num_input_chan, out_filters: int, x_stride, is_training: bool, weights, reuse: bool, scope: str, layer_id: int):
      return [
        MicroChild.Operator.SeparableConv3x3,
        MicroChild.Operator.SeparableConv5x5,
        MicroChild.Operator.AveragePooling,
        MicroChild.Operator.MaxPooling,
        MicroChild.Operator.Identity][op_id](child, num_input_chan, out_filters, x_stride, is_training, weights, reuse, scope, layer_id)


  # Because __call__ is overridden, this superclass is just for ease of find.
  class FixedLayer(LayeredModel):
    """
    Args:
      prev_layers: cache of previous layers. for skip connections
      is_training: for batch_norm
    Returns:
      Output layer
      Number of output channels
    """
    def __init__(self, child, layer_id, arc, hw, c, out_filters, stride, is_training, weights, reuse):
      self.cs = MicroChild.CalibrateSize(child, hw, c, out_filters, is_training, weights, reuse)
      with fw.name_scope("layer_base") as scope:
        self.lb = MicroChild.LayerBase(weights, reuse, scope, out_filters, out_filters, is_training, child.data_format)
      self.x_op = {}
      self.y_op = {}
      ops = []
      used = np.zeros([child.num_cells + 2], dtype=np.int32)
      for cell_id in range(child.num_cells):
        with fw.name_scope(f'cell_{cell_id}') as scope:
          x_id = arc[4 * cell_id]
          used[x_id] += 1
          x_op = arc[4 * cell_id + 1]
          x_stride = stride if x_id in [0, 1] else 1
          with fw.name_scope('x_conv') as scope:
            self.x_op[cell_id] = MicroChild.Operator.new(x_op, child, out_filters, out_filters, x_stride, is_training, weights, reuse, scope, layer_id)
          y_id = arc[4 * cell_id + 2]
          used[y_id] += 1
          y_op = arc[4 * cell_id + 3]
          y_stride = stride if y_id in [0, 1] else 1
          with fw.name_scope('y_conv') as scope:
            self.y_op[cell_id] = MicroChild.Operator.new(y_op, child, out_filters, out_filters, y_stride, is_training, weights, reuse, scope, layer_id)
        ops.append([x_op, y_op])
      c = [out_filters] * (child.num_cells + 2)
      hws = []
      for i in range(child.num_cells + 2):
        if [0, 0] == ops[0]: # Not sure about this check :-\
          hws.append(hw[1] // 2)
        else:
          hws.append(hw[1])
      uhws = []
      for i in range(child.num_cells + 2):
        if used[i] == 0:
          if [0, 0] == ops[0]:  # Not sure about this check :-\
            uhws.append(hw[1] // 2)
          else:
            uhws.append(hw[1])
      out_hw = min(uhws)
      self.fc = MicroChild.FixedCombine(child, used, c, hws, out_hw, out_filters, is_training, weights, reuse)
      self.out_chan = out_filters // 2 * 2 * len(uhws)

      def layer2(layers):
        for cell_id in range(child.num_cells):
          with fw.name_scope(f'cell_{cell_id}') as scope:
            x_id = arc[4 * cell_id]
            x = layers[x_id]
            with fw.name_scope("x_conv") as scope:
              op = self.x_op[cell_id]
              x = op(x)
            y_id = arc[4 * cell_id + 2]
            y = layers[y_id]
            with fw.name_scope("y_conv") as scope:
              op = self.y_op[cell_id]
              y = op(y)

            out = x + y
            layers.append(out)
        return layers

      self.l2 = layer2

    def __call__(self, prev_layers):
      assert 2 == len(prev_layers)
      layers = self.cs(prev_layers)
      with fw.name_scope("layer_base") as scope:
        layers[1] = self.lb(layers[1])
      layers = self.l2(layers)
      return self.fc(layers)


  class MaxPool(LayeredModel):
    def __init__(self, data_format, num_input_chan: int, out_filters: int, reuse: bool, scope: str, curr_cell, prev_cell: int, weights):
      self.layers = [
        fw.max_pool2d([3, 3], [1, 1], 'SAME', data_format=data_format.actual)]
      if num_input_chan != out_filters:
        w = weights.get(
          reuse,
          scope,
          'w',
          [curr_cell + 1, num_input_chan * out_filters],
          None)
        with fw.name_scope('conv'):
          bn = BatchNorm(True, data_format, weights, out_filters, reuse)
          conv2d = lambda x: fw.conv2d(x, fw.reshape(w[prev_cell], [1, 1, num_input_chan, out_filters]), strides=[1, 1, 1, 1], padding='SAME', data_format=data_format.name)
        self.layers += [fw.relu, conv2d, bn]


  class AvgPool(LayeredModel):
    def __init__(self, data_format, num_input_chan: int, out_filters: int, reuse: bool, scope: str, curr_cell, prev_cell: int, weights):
      self.layers = [
        fw.avg_pool2d([3, 3], [1, 1], 'SAME', data_format=data_format.actual)]
      if num_input_chan != out_filters:
        w = weights.get(
          reuse,
          scope,
          'w',
          [curr_cell + 1, num_input_chan * out_filters],
          None)
        with fw.name_scope("conv"):
          bn = BatchNorm(True, data_format, weights, out_filters, reuse)

        def inner(x):
          with fw.name_scope("conv"):
            return bn(
              fw.conv2d(
                fw.relu(x),
                fw.reshape(
                  w[prev_cell],
                  [1, 1, num_input_chan, out_filters]),
                strides=[1, 1, 1, 1],
                padding="SAME",
                data_format=data_format.name))
        self.layers.append(inner)


  class ENASCellInner(LayeredModel):
    def __init__(self, data_format, num_input_chan: int, out_filters: int, weights, reuse: bool, scope: str, num_possible_inputs: int, prev_cell: int):
      self.layers = []
      if num_input_chan != out_filters:
        w = weights.get(
          reuse,
          scope,
          "w",
          [num_possible_inputs, num_input_chan * out_filters],
          None)
        conv2d = lambda x: fw.conv2d(x, fw.reshape(w[prev_cell], [1, 1, num_input_chan, out_filters]), strides=[1, 1, 1, 1], padding='SAME', data_format=data_format.name)
        bn = BatchNorm(True, data_format, weights, out_filters, reuse)
        self.layers += [fw.relu, conv2d, bn]
      else:
        self.layers.append(lambda x: x)


  # Because __call__ is overridden, this superclass is just for ease of find.
  class ENASCell(LayeredModel):
    """Performs an enas operation specified by op_id."""
    def __init__(self, child, curr_cell, prev_cell, num_input_chan: int, out_filters: int, weights, reuse: bool):
      with fw.name_scope("avg_pool") as scope:
        self.ap = MicroChild.AvgPool(child.data_format, num_input_chan, out_filters, reuse, scope, curr_cell, prev_cell, weights)
      with fw.name_scope("max_pool") as scope:
        self.mp = MicroChild.MaxPool(child.data_format, num_input_chan, out_filters, reuse, scope, curr_cell, prev_cell, weights)
      with fw.name_scope("x_conv") as scope:
        self.ec = MicroChild.ENASCellInner(child.data_format, num_input_chan, out_filters, weights, reuse, scope, curr_cell + 1, prev_cell)
      self.ec3 = MicroChild.ENASConvOuter(child, curr_cell, prev_cell, 3, out_filters, weights, reuse)
      self.ec5 = MicroChild.ENASConvOuter(child, curr_cell, prev_cell, 5, out_filters, weights, reuse)

    def __call__(self, x, op_id):
      with fw.name_scope("avg_pool"):
        avg_pool = self.ap(x)
      with fw.name_scope("max_pool"):
        max_pool = self.mp(x)
      with fw.name_scope("x_conv"):
        x = self.ec(x)
      return fw.stack([self.ec3(x), self.ec5(x), avg_pool, max_pool, x], axis=0)[op_id, :, :, :, :]


  class ENASConvInner(LayeredModel):
    def __init__(self, weights, reuse: bool, scope: str, num_possible_inputs: int, filter_size: int, prev_cell: int, out_filters: int, data_format):
      with fw.name_scope("bn") as scope1:
        offset = weights.get(
          reuse,
          scope1,
          "offset", [num_possible_inputs, out_filters],
          fw.zeros_init())[prev_cell]
        scale = weights.get(
          reuse,
          scope1,
          "scale", [num_possible_inputs, out_filters],
          fw.ones_init())[prev_cell]
      weight_depthwise = weights.get(
        reuse,
        scope,
        "w_depth",
        [num_possible_inputs, filter_size * filter_size * out_filters], None)[prev_cell, :]
      weight_pointwise = weights.get(
        reuse,
        scope,
        "w_point",
        [num_possible_inputs, out_filters * out_filters], None)[prev_cell, :]
      def inner(x):
        return fw.fused_batch_norm(
          x=fw.separable_conv2d(
            fw.relu(x),
            depthwise_filter=fw.reshape(
              weight_depthwise,
              [filter_size, filter_size, out_filters, 1]),
            pointwise_filter=fw.reshape(
              weight_pointwise,
              [1, 1, out_filters, out_filters]),
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format=data_format.name),
          scale=scale,
          offset=offset,
          mean=x,
          variance=x,
          epsilon=1e-5,
          data_format=data_format.name,
          is_training=True)[0]

      self.layers = [inner]


  class ENASConvOuter(LayeredModel):
    def __init__(self, child, curr_cell, prev_cell, filter_size, out_filters: int, weights, reuse: bool, stack_conv=2):
      self.layers = []
      with fw.name_scope("conv_{0}x{0}".format(filter_size)) as scope:
        for conv_id in range(stack_conv):
          with fw.name_scope("stack_{0}".format(conv_id)) as scope:
            # create params and pick the correct path
            self.layers.append(MicroChild.ENASConvInner(weights, reuse, scope, curr_cell + 2, filter_size, prev_cell, out_filters, child.data_format))


  class ENASLayerInner(LayeredModel):
    def __init__(self, weights, reuse: bool, scope:str, num_cells: int, out_filters: int, data_format):
      self.stack = lambda x: fw.stack(x, axis=0)
      self.gather = lambda x, indices: fw.gather(x, indices, axis=0)
      self.micro_enas = lambda x, num_outs, prev_layers0: data_format.micro_enas(
        x,
        prev_layers0,
        num_outs,
        out_filters)
      w = weights.get(
        reuse,
        scope,
        "w",
        [num_cells + 2, out_filters * out_filters],
        None)
      filters = lambda indices, num_outs: fw.reshape(
        fw.gather(
          w,
          indices,
          axis=0),
        [1, 1, num_outs * out_filters, out_filters])
      self.conv2d = lambda x, indices, num_outs: fw.conv2d(
        x,
        filters(indices, num_outs),
        strides=[1, 1, 1, 1],
        padding='SAME',
        data_format = data_format.name)
      self.bn = BatchNorm(True, data_format, weights, out_filters, reuse)
      self.reshape = lambda x, prev_layers_0: fw.reshape(x, fw.shape(prev_layers_0))

    def __call__(self, x, indices, num_outs, prev_layers_0):
      return self.reshape(
        self.bn(
          self.conv2d(
            fw.relu(
              self.micro_enas(
                self.gather(self.stack(x), indices),
                num_outs,
                prev_layers_0)),
            indices,
          num_outs)),
        prev_layers_0)


    class Indices(LayeredModel):
      def __init__(self):
        self.layers = [
          fw.add_n,
          lambda x: fw.equal(x, 0),
          fw.where,
          fw.to_int32,
          lambda x: fw.reshape(x, [-1])]


  class ENASLayer(object):
    """
    Args:
      layer_id: current layer
      prev_layers: cache of previous layers. for skip connections
      start_idx: where to start looking at. technically, we can infer this
        from layer_id, but why bother...
    Output channels: out_filters
    """
    def __init__(self, child, arc, hw, c, out_filters, weights, reuse):
      self.cs = MicroChild.CalibrateSize(child, hw, c, out_filters, True, weights, reuse)
      ec_x = {}
      ec_y = {}
      for cell_id in range(child.num_cells):
          with fw.name_scope(f"cell_{cell_id}"):
            with fw.name_scope("x"):
              x_id = arc[4 * cell_id] # always in {0, 1}
              ec_x[cell_id] = MicroChild.ENASCell(child, cell_id, x_id, out_filters, out_filters, weights, reuse)
            with fw.name_scope("y"):
              y_id = arc[4 * cell_id + 2]
              ec_y[cell_id] = MicroChild.ENASCell(child, cell_id, y_id, out_filters, out_filters, weights, reuse)

      def l0(layers):
        used = []
        for cell_id in range(child.num_cells):
          prev_layers = fw.stack(layers, axis=0) # Always 2, N, H, W, C or 2, N, C, H, W
          with fw.name_scope(f"cell_{cell_id}"):
            with fw.name_scope("x"):
              x_id = arc[4 * cell_id] # always in {0, 1}
              x = prev_layers[x_id, :, :, :, :]
              x = ec_x[cell_id](x, arc[4 * cell_id + 1])
              x_used = fw.one_hot(x_id, depth=child.num_cells + 2, dtype=fw.int32)
            with fw.name_scope("y"):
              y_id = arc[4 * cell_id + 2]
              y = prev_layers[y_id, :, :, :, :]
              y = ec_y[cell_id](y, arc[4 * cell_id + 3])
              y_used = fw.one_hot(y_id, depth=child.num_cells + 2, dtype=fw.int32)
            out = x + y
            used.extend([x_used, y_used])
            layers.append(out)
        return prev_layers, layers, used

      self.l0 = l0
      with fw.name_scope("final_conv") as scope:
        el = MicroChild.ENASLayerInner(
          weights,
          reuse,
          scope,
          child.num_cells,
          out_filters,
          child.data_format)

      def indices(layers, prev_layers, used):
        i = MicroChild.ENASLayerInner.Indices()
        ixs = i(used)
        num_outs = fw.size(ixs)
        return el(layers, ixs, num_outs, prev_layers[0])

      self.indices = indices

    def __call__(self, prev_layers):
      assert 2 == len(prev_layers), "Need exactly two inputs."
      layers = self.cs(prev_layers)
      prev_layers, layers, used = self.l0(layers)
      return self.indices(layers, prev_layers, used)


  class Loss(LayeredModel):
    def __init__(self, y):
      self.layers = [
        lambda x: fw.sparse_softmax_cross_entropy_with_logits(logits=x, labels=y),
        fw.reduce_mean]


  class Accuracy(LayeredModel):
    def __init__(self, y):
      self.layers = [
        lambda x: fw.argmax(x, axis=1),
        fw.to_int32,
        lambda x: fw.equal(x, y),
        fw.to_int32,
        fw.reduce_sum]

  # override
  def _build_train(self, y):
    print("-" * 80)
    print("Build train graph")
    logit_fn1 = MicroChild.Loss(y)
    loss = lambda logits_aux_logits: logit_fn1(logits_aux_logits[0])
    if self.use_aux_heads:
      self.aux_loss = lambda logits_aux_logits: fw.reduce_mean(fw.sparse_softmax_cross_entropy_with_logits(
        logits=logits_aux_logits[0], labels=y))
      train_loss = lambda logits_aux_logits: loss(logits_aux_logits) + 0.4 * self.aux_loss(logits_aux_logits)
    else:
      train_loss = loss
    train_op, lr, grad_norm, optimizer = get_train_ops(
      self.global_step,
      self.learning_rate,
      clip_mode=self.clip_mode,
      l2_reg=self.l2_reg,
      num_train_batches=self.num_train_batches,
      optim_algo=self.optim_algo)

    logit_fn2 = MicroChild.Accuracy(y)
    train_acc = lambda logits_aux_logits: logit_fn2(logits_aux_logits[0])

    return loss, train_loss, train_acc, train_op, lr, grad_norm, optimizer


  # override
  def _build_valid(self, y):
    print("-" * 80)
    print("Build valid graph")
    predictions = lambda logits: fw.to_int32(fw.argmax(logits, axis=1))
    retval = (
      predictions,
      lambda logits: fw.reduce_sum(fw.to_int32(fw.equal(predictions(logits), y))))
    return retval


  # override
  def _build_test(self, y):
    print("-" * 80)
    print("Build test graph")
    predictions = lambda logits: fw.to_int32(fw.argmax(logits, axis=1))
    return (
      predictions,
      lambda logits: fw.reduce_sum(fw.to_int32(fw.equal(predictions(logits), y))))


  class ValidationRLShuffle(LayeredModel):
    def __init__(self, child, shuffle):
      with fw.device('/cpu:0'):
        # shuffled valid data: for choosing validation model
        if not shuffle:
          self.layers = [lambda x: child.data_format.child_init(x)] # 0
        else:
          self.layers = [lambda x: x] # 0
        self.layers.append(lambda x, y: fw.shuffle_batch((x, y), child.batch_size, child.seed, 25000)) # 1
        if shuffle:

          def _pre_process2(image):
              return child.data_format.child_init_preprocess(
                  fw.random_flip_left_right(
                      fw.random_crop(
                          fw.pad(image, [[4, 4], [4, 4], [0, 0]]),
                          [32, 32, 3],
                          seed=child.seed),
                      seed=child.seed))

          def _pre_process(image_batch, label_batch):
              return fw.map_fn(_pre_process2, image_batch), label_batch

          def layer2(x):
            return x.map(_pre_process)

          self.layers.append(layer2) # 2
        else:
          self.layers.append(lambda x: x) # 2

    def __call__(self, x, y):
      with fw.device('/cpu:0'):
        x = self.layers[0](x)
        dataset_valid_shuffle = self.layers[1](x, y)
        dataset_valid_shuffle = self.layers[2](dataset_valid_shuffle)
        return dataset_valid_shuffle


  class ValidationRL(LayeredModel):
    def __init__(self):
      with fw.device('/cpu:0'):
        self.layers = [
          lambda logits: fw.argmax(logits, axis=1),
          fw.to_int32,
          lambda x, y: fw.equal(x, y),
          fw.to_int32,
          fw.reduce_sum]

    def __call__(self, logits, y_valid_shuffle):
      with fw.device('/cpu:0'):
        valid_shuffle_preds = self.layers[0](logits)
        valid_shuffle_preds = self.layers[1](valid_shuffle_preds)
        valid_shuffle_acc =   self.layers[2](valid_shuffle_preds, y_valid_shuffle)
        valid_shuffle_acc =   self.layers[3](valid_shuffle_acc)
        return                self.layers[4](valid_shuffle_acc)

  def connect_controller(self, controller_model):
    if self.fixed_arc is None:
      self.current_controller_normal_arc, self.current_controller_reduce_arc = lambda: controller_model.current_normal_arc, lambda: controller_model.current_reduce_arc
    else:
      fixed_arc = np.array([int(x) for x in self.fixed_arc.split(" ") if x])
      self.current_controller_normal_arc = lambda: fixed_arc[:4 * self.num_cells]
      self.current_controller_reduce_arc = lambda: fixed_arc[4 * self.num_cells:]

    self.loss, self.train_loss, self.train_acc, train_op, lr, grad_norm, optimizer = self._build_train(self.dataset)
    self.valid_preds, self.valid_acc = self._build_valid(self.dataset_valid)
    self.test_preds, self.test_acc = self._build_test(self.dataset_test)
    return train_op, lr, grad_norm, optimizer