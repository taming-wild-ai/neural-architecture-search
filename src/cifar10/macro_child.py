from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import src.framework as fw

from src.cifar10.child import Child
from src.cifar10.image_ops import BatchNorm, BatchNormWithMask

from src.utils import count_model_params, get_train_ops, DEFINE_integer, LayeredModel

DEFINE_integer("child_num_branches", 6, "")
DEFINE_integer("child_out_filters_scale", 1, "")


class MacroChild(Child):
  def __init__(self,
               images,
               labels,
               clip_mode=None,
               lr_dec_start=0,
               lr_min=None,
               optim_algo=None,
               name="child",
              ):
    super(self.__class__, self).__init__(
      images,
      labels,
      clip_mode=clip_mode,
      lr_dec_start=lr_dec_start,
      optim_algo=optim_algo,
      name=name)
    FLAGS = fw.FLAGS
    self.whole_channels = FLAGS.controller_search_whole_channels
    self.lr_min = lr_min
    self.out_filters_scale = FLAGS.child_out_filters_scale
    self.out_filters = FLAGS.child_out_filters * self.out_filters_scale

    self.num_branches = FLAGS.child_num_branches

    pool_distance = self.num_layers // 3
    self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]


  class SkipPath(LayeredModel):
    def __init__(self, stride_spec, data_format, weights, reuse: bool, scope: str, num_input_chan: int, out_filters: int):
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
          "SAME", # Only difference from MicroChild.SkipPath
          data_format=data_format.name)
      self.layers = [avg_pool, conv2d]


  class Dropout(LayeredModel):
    def __init__(self, data_format, is_training, keep_prob, weights, reuse, scope: str, num_inp_chan: int):
      w = weights.get(
        reuse,
        scope,
        "w",
        [num_inp_chan, 10],
        None)
      def matmul(x):
        return fw.matmul(x, w)
      self.layers = [data_format.global_avg_pool]
      if is_training:
        self.layers += [lambda x: fw.dropout(x, keep_prob)]
      self.layers += [matmul]


  # Because __call__ is overridden, this superclass is just for ease of search.
  class Model(LayeredModel):
    def __init__(self, child, is_training: bool, reuse=False):
      self.child = child
      self.enas_layers = []
      self.model_factorized_reduction = {}
      with fw.name_scope(child.name):
        with fw.name_scope("stem_conv") as scope:
          self.model_layers = [Child.StemConv(child.weights, reuse, scope, child.out_filters, is_training, child.data_format)]

        out_filters = child.out_filters
        layers_channels = [out_filters]

        if child.whole_channels:
          start_idx = 0
        else:
          start_idx = self.child.num_branches

        for layer_id in range(child.num_layers):
          with fw.name_scope("layer_{0}".format(layer_id)):
            if child.fixed_arc is None:
              with fw.name_scope("conv_1") as scope:
                input_conv4 = Child.InputConv(
                  child.weights,
                  reuse,
                  scope,
                  1,
                  out_filters,
                  child.out_filters,
                  is_training,
                  child.data_format)
                input_conv5 = Child.InputConv(
                  child.weights,
                  reuse,
                  scope,
                  1,
                  out_filters,
                  child.out_filters,
                  is_training,
                  child.data_format)
              self.model_layers.append(MacroChild.ENASLayer(child, layer_id, start_idx, out_filters, out_filters, is_training, child.weights, reuse, input_conv4, input_conv5))
              layers_channels.append(out_filters)
            else:
              self.model_layers.append(MacroChild.FixedLayer(child, layer_id, start_idx, out_filters, out_filters, is_training, child.weights, reuse))
              layers_channels.append(out_filters)
            if layer_id in child.pool_layers:
              if child.fixed_arc is not None:
                out_filters *= 2
              with fw.name_scope("pool_at_{0}".format(layer_id)):
                pooled_layers_channels = []
                for i, layer in enumerate(self.model_layers):
                  with fw.name_scope("from_{0}".format(i)):
                    self.model_factorized_reduction[(layer_id, i)] = Child.FactorizedReduction(child, layers_channels[i], out_filters, 2, is_training, child.weights, reuse)
                  pooled_layers_channels.append(out_filters)
                layers_channels = pooled_layers_channels
          if child.whole_channels:
            start_idx += 1 + layer_id
          else:
            start_idx += 2 * child.num_branches + layer_id

        with fw.name_scope("fc") as scope:
          self.model_dropout = MacroChild.Dropout(
            child.data_format,
            is_training,
            child.keep_prob,
            child.weights,
            reuse,
            scope,
            layers_channels[-1])
        print("Model has {0} params".format(count_model_params(child.tf_variables())))

    def __call__(self, images):
      with fw.name_scope(self.child.name):
        layers = [self.model_layers[0](images)]

        if self.child.whole_channels:
          start_idx = 0
        else:
          start_idx = self.child.num_branches

        for layer_id in range(self.child.num_layers):
          with fw.name_scope("layer_{0}".format(layer_id)):
            if self.child.fixed_arc is None:
              x = self.model_layers[layer_id + 1](layers)
              layers.append(x)
            else:
              x = self.model_layers[layer_id + 1](layers)
              layers.append(x)
            if layer_id in self.child.pool_layers:
              with fw.name_scope("pool_at_{0}".format(layer_id)):
                pooled_layers = []
                for i, layer in enumerate(layers):
                  with fw.name_scope("from_{0}".format(i)):
                    x = self.model_factorized_reduction[(layer_id, i)](layer)
                  pooled_layers.append(x)
                layers = pooled_layers
          if self.child.whole_channels:
            start_idx += 1 + layer_id
          else:
            start_idx += 2 * self.child.num_branches + layer_id
          print(layers[-1])

        with fw.name_scope("fc") as scope:
          x = self.model_dropout(x)
      return x


  class FinalConv(LayeredModel):
    def __init__(self, num_branches: int, count, weights, reuse, scope: str, out_filters: int, is_training: bool, data_format):
      w_mask = fw.constant([False] * (num_branches * out_filters), fw.bool)
      new_range = fw.range(0, num_branches * out_filters, dtype=fw.int32)
      for i in range(num_branches):
        start = out_filters * i + count[2 * i]
        w_mask = fw.logical_or(w_mask, fw.logical_and(
          start <= new_range, new_range < start + count[2 * i + 1]))
      w = weights.get(
        reuse,
        scope,
        "w",
        [num_branches * out_filters, out_filters],
        None)
      def conv2d(x):
        return fw.conv2d(
          x,
          fw.reshape(
            fw.boolean_mask(
              w,
              w_mask),
              [1, 1, -1, out_filters]),
          [1, 1, 1, 1],
          'SAME',
          data_format=data_format.name)
      bn = BatchNorm(is_training, data_format, weights, out_filters, reuse)
      self.layers = [conv2d, bn, fw.relu]


  # Because __call__ is overridden, this superclass is just for ease of find.
  class ENASLayerNotWholeChannels(LayeredModel):
    def __init__(self, child, layer_id, start_idx, num_input_chan, out_filters, is_training, weights, reuse, input_conv_avg, input_conv_max):
      count = child.sample_arc[start_idx:start_idx + 2 * child.num_branches]
      branch_0 = MacroChild.ConvBranch(child, 3, is_training, count[1], num_input_chan, out_filters, weights, reuse, 1, count[0], False)
      branch_1 = MacroChild.ConvBranch(child, 3, is_training, count[3], num_input_chan, out_filters, weights, reuse, 1, count[2], True)
      branch_2 = MacroChild.ConvBranch(child, 5, is_training, count[5], num_input_chan, out_filters, weights, reuse, 1, count[4], False)
      branch_3 = MacroChild.ConvBranch(child, 5, is_training, count[7], num_input_chan, out_filters, weights, reuse, 1, count[6], True)
      branch_4 = MacroChild.PoolBranch(child, count[9], "avg", input_conv_avg, count[8])
      branch_5 = MacroChild.PoolBranch(child, count[11], "max", input_conv_max, count[10])
      with fw.name_scope("final_conv") as scope:
        final_conv = MacroChild.FinalConv(
              child.num_branches,
              count,
              weights,
              reuse,
              scope,
              out_filters,
              is_training,
              child.data_format)

      def branches(inputs):
        branches = []
        with fw.name_scope("branch_0"):
          branches.append(branch_0(inputs))
        with fw.name_scope("branch_1"):
          branches.append(branch_1(inputs))
        with fw.name_scope("branch_2"):
          branches.append(branch_2(inputs))
        with fw.name_scope("branch_3"):
          branches.append(branch_3(inputs))
        if child.num_branches >= 5:
          with fw.name_scope("branch_4"):
            branches.append(branch_4(inputs))
        if child.num_branches >= 6:
          with fw.name_scope("branch_5"):
            branches.append(branch_5(inputs))
        with fw.name_scope("final_conv"):
          branches = child.data_format.enas_layer(inputs, branches)
          return final_conv(branches)

      self.layers = [branches]
      if layer_id > 0:
        self.has_skip_layer = True
        skip_start = start_idx + 1
        skip = child.sample_arc[skip_start: skip_start + layer_id]
        with fw.name_scope("skip"):
          self.layers.append(MacroChild.ENASSkipLayers(layer_id, skip, is_training, child.data_format, weights, num_input_chan, reuse))
      else:
        self.has_skip_layer = False

    def __call__(self, prev_layers):
      inputs = prev_layers[-1]
      out = self.layers[0](inputs)
      if self.has_skip_layer:
        return self.layers[1](out, prev_layers)
      else:
        return out


  # Because __call__ is overridden, this superclass is just for ease of find.
  class ENASLayerWholeChannels(LayeredModel):
    def __init__(self, child, layer_id, start_idx, num_input_chan, out_filters, is_training, weights, reuse, input_conv_avg, input_conv_max):
      self.layers = [lambda inputs: child.data_format.get_HW(inputs)]
      count = child.sample_arc[start_idx]
      branch_0 = MacroChild.ConvBranch(child, 3, is_training, out_filters, num_input_chan, out_filters, weights, reuse, 1, 0, False)
      branch_1 = MacroChild.ConvBranch(child, 3, is_training, out_filters, num_input_chan, out_filters, weights, reuse, 1, 0, True)
      branch_2 = MacroChild.ConvBranch(child, 5, is_training, out_filters, num_input_chan, out_filters, weights, reuse, 1, 0, False)
      branch_3 = MacroChild.ConvBranch(child, 5, is_training, out_filters, num_input_chan, out_filters, weights, reuse, 1, 0, True)
      branch_4 = MacroChild.PoolBranch(child, out_filters, "avg", input_conv_avg, 0)
      branch_5 = MacroChild.PoolBranch(child, out_filters, "max", input_conv_max, 0)
      def branches(inputs):
        arms = {}
        with fw.name_scope("branch_0"):
          y = branch_0(inputs)
          arms[fw.equal(count, 0)] = lambda: y
        with fw.name_scope("branch_1"):
          y = branch_1(inputs)
          arms[fw.equal(count, 1)] = lambda: y
        with fw.name_scope("branch_2"):
          y = branch_2(inputs)
          arms[fw.equal(count, 2)] = lambda: y
        with fw.name_scope("branch_3"):
          y = branch_3(inputs)
          arms[fw.equal(count, 3)] = lambda: y
        with fw.name_scope("branch_4"):
          y = branch_4(inputs)
          arms[fw.equal(count, 4)] = lambda: y
        with fw.name_scope("branch_5"):
          y = branch_5(inputs)
          arms[fw.equal(count, 5)] = lambda: y
        return fw.case(
          arms,
          default=lambda: fw.constant(0, fw.float32),
          exclusive=True)

      self.layers.append(branches)

      def reshape(inputs, h, w):
        child.data_format.set_shape(inputs, h, w, out_filters)
        return inputs

      self.layers.append(reshape)
      if layer_id > 0:
        self.has_skip_layer = True
        skip_start = start_idx + 1
        skip = child.sample_arc[skip_start: skip_start + layer_id]
        with fw.name_scope("skip"):
          self.layers.append(MacroChild.ENASSkipLayers(layer_id, skip, is_training, child.data_format, weights, num_input_chan, reuse))
      else:
        self.has_skip_layer = False

    def __call__(self, prev_layers):
      inputs = prev_layers[-1]
      inp_h, inp_w = self.layers[0](inputs)
      out = self.layers[1](inputs)
      out = self.layers[2](out, inp_h, inp_w)
      if self.has_skip_layer:
        return self.layers[3](out, prev_layers)
      else:
        return out


  def ENASLayer(self, layer_id, start_idx, num_input_chan: int, out_filters: int, is_training: bool, weights, reuse: bool, input_conv_avg, input_conv_max):
    """
    Args:
      layer_id: current layer
      prev_layers: cache of previous layers. for skip connections
      start_idx: where to start looking at. technically, we can infer this
        from layer_id, but why bother...
      is_training: for batch_norm
    """
    if self.whole_channels:
      return MacroChild.ENASLayerWholeChannels(self, layer_id, start_idx, num_input_chan, out_filters, is_training, weights, reuse, input_conv_avg, input_conv_max)
    else:
      return MacroChild.ENASLayerNotWholeChannels(self, layer_id, start_idx, num_input_chan, out_filters, is_training, weights, reuse, input_conv_avg, input_conv_max)


  # Because __call__ is overridden, this superclass is just for ease of find.
  class ENASSkipLayers(LayeredModel):
    def __init__(self, layer_id, skip, is_training, data_format, weights, num_input_chan, reuse: bool):
      def skip_connections(prev_layers):
        res_layers = []
        for i in range(layer_id):
          res_layers.append(
            fw.cond(
              fw.equal(skip[i], 1),
              lambda: prev_layers[i],
              lambda: fw.zeros_like(prev_layers[i])))
        return res_layers

      def append_input(res_layers, x):
        res_layers.append(x)
        return res_layers

      self.layers = [
        skip_connections,
        append_input,
        lambda res_layers: fw.add_n(res_layers),
        BatchNorm(is_training, data_format, weights, num_input_chan, reuse)]

    def __call__(self, x, prev_layers):
      res_layers = self.layers[0](prev_layers)
      res_layers = self.layers[1](res_layers, x)
      x =          self.layers[2](res_layers)
      return       self.layers[3](x)


  # Because __call__ is overridden, this superclass is just for ease of find.
  class FixedLayerNotWholeChannels(LayeredModel):
    def __init__(self, child, layer_id, start_idx, num_input_chan: int, out_filters: int, is_training: bool, weights, reuse: bool):
      count = (
        child.sample_arc[start_idx:start_idx + 2 * child.num_branches]
        * child.out_filters_scale)
      total_out_channels = 0
      with fw.name_scope("branch_0"):
        total_out_channels += count[1]
        branch_0 = MacroChild.ConvBranch(child, 3, is_training, count[1], num_input_chan, out_filters, weights, reuse, 1, count[1], False)
      with fw.name_scope("branch_1"):
        total_out_channels += count[3]
        branch_1 = MacroChild.ConvBranch(child, 3, is_training, count[3], num_input_chan, out_filters, weights, reuse, 1, count[3], True)
      with fw.name_scope("branch_2"):
        total_out_channels += count[5]
        branch_2 = MacroChild.ConvBranch(child, 5, is_training, count[5], num_input_chan, out_filters, weights, reuse, 1, count[5], False)
      with fw.name_scope('branch_3'):
        total_out_channels += count[7]
        branch_3 = MacroChild.ConvBranch(child, 5, is_training, count[7], num_input_chan, out_filters, weights, reuse, 1, count[7], True)
      if child.num_branches >= 5:
        with fw.name_scope("conv_1") as scope:
          input_conv4 = Child.InputConv(
            weights,
            reuse,
            scope,
            1,
            num_input_chan,
            child.out_filters,
            is_training,
            child.data_format)
        with fw.name_scope("branch_4"):
          total_out_channels += count[9]
          branch_4 = MacroChild.PoolBranch(self, count[9], "avg", input_conv4)
      if child.num_branches >= 6:
        with fw.name_scope("conv_1") as scope:
          input_conv5 = Child.InputConv(
            weights,
            reuse,
            scope,
            1,
            num_input_chan,
            child.out_filters,
            is_training,
            child.data_format)
        with fw.name_scope("branch_5"):
          total_out_channels += count[11]
          branch_5 = MacroChild.PoolBranch(child, count[11], "max", input_conv5)
      with fw.name_scope("final_conv") as scope:
        final_conv = MacroChild.Conv1x1(weights, reuse, scope, total_out_channels, out_filters, is_training, child.data_format)

      def branches(inputs):
        branches = []
        with fw.name_scope("branch_0"):
          branches.append(branch_0(inputs))
        with fw.name_scope("branch_1"):
          branches.append(branch_1(inputs))
        with fw.name_scope("branch_2"):
          branches.append(branch_2(inputs))
        with fw.name_scope("branch_3"):
          branches.append(branch_3(inputs))
        if child.num_branches >= 5:
          with fw.name_scope("branch_4"):
            branches.append(branch_4(inputs))
        if child.num_branches >= 6:
          with fw.name_scope("branch_5"):
            branches.append(branch_5(inputs))
          with fw.name_scope("final_conv") as scope:
            branches = child.data_format.fixed_layer(branches)
            return final_conv(branches)

      self.layers = [branches]

      if layer_id > 0:
        self.has_skip_layer = True
        skip_start = start_idx + 2 * child.num_branches
        skip = child.sample_arc[skip_start: skip_start + layer_id]
        total_skip_channels = np.sum(skip) + 1
        with fw.name_scope("skip") as scope:
          conv1x1 = MacroChild.Conv1x1(
            weights,
            reuse,
            scope,
            total_skip_channels * out_filters,
            out_filters,
            is_training,
            child.data_format)

        def skip_connections(x, prev_layers):
          res_layers = []
          for i in range(layer_id):
            if skip[i] == 1:
              res_layers.append(prev_layers[i])
          prev = res_layers + [x]
          prev = child.data_format.fixed_layer(prev)
          with fw.name_scope("skip") as scope:
            return conv1x1(prev)

        self.layers.append(skip_connections)
      else:
        self.has_skip_layer = False

    def __call__(self, prev_layers):
      inputs = prev_layers[-1]
      out = self.layers[0](inputs)
      if self.has_skip_layer:
        return self.layers[1](out, prev_layers)
      else:
        return out

  # Because __call__ is overridden, this superclass is just for ease of find.
  class FixedLayerWholeChannels(LayeredModel):
    def __init__(self, child, layer_id, start_idx, num_input_chan: int, out_filters: int, is_training: bool, weights, reuse: bool):
      self.layers = []
      inp_c = num_input_chan
      count = child.sample_arc[start_idx]
      if count in [0, 1, 2, 3]:
        filter_size = [3, 3, 5, 5][count]
        with fw.name_scope("conv_1x1") as scope:
          self.layers.append(Child.Conv1x1(weights, reuse, scope, inp_c, out_filters, is_training, child.data_format))
        with fw.name_scope(f"conv_{filter_size}x{filter_size}") as scope:
          self.layers.append(Child.ConvNxN(weights, reuse ,scope, filter_size, out_filters, child.data_format, is_training))
      elif count == 4:
        pass # TODO Fill this in
      elif count == 5:
        pass # TODO Fill this in
      else:
        raise ValueError(f"Unknown operation number '{count}'")
      if layer_id > 0:
        self.has_skip_layer = True
        skip_start = start_idx + 1
        skip = child.sample_arc[skip_start: skip_start + layer_id]
        total_skip_channels = np.sum(skip) + 1
        with fw.name_scope("skip") as scope:
          conv1x1 = MacroChild.Conv1x1(
            weights,
            reuse,
            scope,
            total_skip_channels * out_filters,
            out_filters,
            is_training,
            child.data_format)

        def skip_connections(x, prev_layers):
          res_layers = []
          for i in range(layer_id):
            if skip[i] == 1:
              res_layers.append(prev_layers[i])
          prev = res_layers + [x]
          prev = child.data_format.fixed_layer(prev)
          with fw.name_scope("skip"):
            return conv1x1(prev)

        self.layers.append(skip_connections)
      else:
        self.has_skip_layer = False

    def __call__(self, prev_layers):
      inputs = prev_layers[-1]
      out = self.layers[0](inputs)
      out = self.layers[1](out)
      if self.has_skip_layer:
        return self.layers[2](out, prev_layers)
      else:
        return out


  def FixedLayer(
      self, layer_id, start_idx, num_input_chan: int, out_filters: int, is_training: bool, weights, reuse: bool):
    """
    Args:
      layer_id: current layer
      prev_layers: cache of previous layers. for skip connections
      start_idx: where to start looking at. technically, we can infer this
        from layer_id, but why bother...
      is_training: for batch_norm
    """
    if self.whole_channels:
      return MacroChild.FixedLayerWholeChannels(self, layer_id, start_idx, num_input_chan, out_filters, is_training, weights, reuse)
    else:
      return MacroChild.FixedLayerNotWholeChannels(self, layer_id, start_idx, num_input_chan, out_filters, is_training, weights, reuse)


  class OutConv(LayeredModel):
    def __init__(self, weights, reuse: bool, scope: str, filter_size: int, inp_c: int, count: int, data_format, is_training: bool):
      w = weights.get(
        reuse,
        scope,
        "w",
        [filter_size, filter_size, inp_c, count],
        None)
      def conv2d(x):
        return fw.conv2d(
          x,
          w,
          [1, 1, 1, 1],
          "SAME",
          data_format=data_format.name)
      bn = BatchNorm(is_training, data_format, weights, filter_size, reuse)
      self.layers = [conv2d, bn, fw.relu]


  class SeparableConv(LayeredModel):
    def __init__(self, weights, reuse: bool, scope: str, filter_size: int, out_filters: int, ch_mul: int, count: int, data_format, is_training: bool):
      w_depth = weights.get(
        reuse,
        scope,
        "w_depth",
        [filter_size, filter_size, out_filters, ch_mul],
        None)
      w_point = weights.get(
        reuse,
        scope,
        "w_point",
        [1, 1, out_filters * ch_mul, count],
        None)
      def sep_conv2d(x):
        return fw.separable_conv2d(
          x,
          w_depth,
          w_point,
          data_format,
          strides=[1, 1, 1, 1],
          padding="SAME",
          data_format=data_format.name)
      bn = BatchNorm(is_training, data_format, weights, out_filters, reuse)
      self.layers = [sep_conv2d, bn, fw.relu]


  class SeparableConvMasked(LayeredModel):
    def __init__(self, weights, reuse, scope, filter_size, out_filters: int, ch_mul: int, start_idx: int, count: int, data_format, is_training: bool):
      self.mask = fw.range(0, out_filters, dtype=fw.int32)
      w_depth = weights.get(
        reuse,
        scope,
        "w_depth",
        [filter_size, filter_size, out_filters, ch_mul],
        None)
      w_point = weights.get(
        reuse,
        scope,
        "w_point",
        [out_filters, out_filters * ch_mul],
        None)
      def sep_conv2d(x):
        return fw.separable_conv2d(
          x,
          w_depth,
          fw.reshape(
            fw.transpose(
              w_point[start_idx:start_idx+count, :],
              [1, 0]),
            [1, 1, out_filters * ch_mul, count]),
          strides=[1, 1, 1, 1],
          padding="SAME",
          data_format=data_format.name)
      self.layers = [
        sep_conv2d,
        BatchNormWithMask(
          is_training,
          fw.logical_and(start_idx <= self.mask, self.mask < start_idx + count),
          out_filters,
          weights,
          reuse,
          data_format=data_format.name),
        fw.relu]


  class OutConvMasked(LayeredModel):
    def __init__(self, weights, reuse, scope, filter_size, out_filters: int, start_idx: int, count: int, data_format, is_training: bool):
      self.mask = fw.range(0, out_filters, dtype=fw.int32)
      w = weights.get(
        reuse,
        scope,
        "w",
        [filter_size, filter_size, out_filters, out_filters],
        None)
      def conv2d(x):
        return fw.conv2d(
          x,
          fw.transpose(
            fw.transpose(
              w,
              [3, 0, 1, 2])[start_idx:start_idx+count, :, :, :],
            [1, 2, 3, 0]),
          [1, 1, 1, 1],
          "SAME",
          data_format=data_format.name)
      self.layers = [
        conv2d,
        BatchNormWithMask(
          is_training,
          fw.logical_and(start_idx <= self.mask, self.mask < start_idx + count),
          out_filters,
          weights,
          reuse,
          data_format=data_format.name),
        fw.relu]


  class ConvBranch(LayeredModel):
    def __init__(self, child, filter_size, is_training: bool, count, num_input_chan: int, out_filters: int, weights, reuse: bool, ch_mul: int, start_idx, separable: bool):
      """
      Args:
        start_idx: where to start taking the output channels. if None, assuming
          fixed_arc mode
        count: how many output_channels to take.
      """
      if start_idx is None:
        assert child.fixed_arc is not None, "you screwed up!"
      with fw.name_scope("inp_conv_1") as scope:
        inp_conv_1 = Child.InputConv(
          weights,
          reuse,
          scope,
          1,
          num_input_chan,
          out_filters,
          is_training,
          child.data_format)
        self.layers = [inp_conv_1]
      with fw.name_scope("out_conv_{}".format(filter_size)) as scope:
        if start_idx is None:
          if separable:
            sep_conv = MacroChild.SeparableConv(
              weights,
              reuse,
              scope,
              filter_size,
              out_filters,
              ch_mul,
              count,
              child.data_format,
              is_training)
            self.layers += [sep_conv]
          else:
            out_conv = MacroChild.OutConv(
              weights,
              reuse,
              scope,
              filter_size,
              num_input_chan,
              count,
              child.data_format,
              is_training)
            self.layers += [out_conv]
        else:
          if separable:
            sep_conv = MacroChild.SeparableConvMasked(
              weights,
              reuse,
              scope,
              filter_size,
              out_filters,
              ch_mul,
              start_idx,
              count,
              child.data_format,
              is_training)
            self.layers += [sep_conv]
          else:
            out_conv = MacroChild.OutConvMasked(
              weights,
              reuse,
              scope,
              filter_size,
              out_filters,
              start_idx,
              count,
              child.data_format,
              is_training)
            self.layers += [out_conv]


  class PoolBranch(LayeredModel):
    def __init__(self, child, count, avg_or_max: str, input_conv, start_idx=None):
      """
      Args:
        start_idx: where to start taking the output channels. if None, assuming
          fixed_arc mode
        count: how many output_channels to take.
      """
      if start_idx is None:
        assert child.fixed_arc is not None, "you screwed up!"

      def layer0(x):
        with fw.name_scope("conv_1"):
          return input_conv(x)

      self.layers = [layer0]

      def avg_pool2d(x):
        with fw.name_scope("pool"):
          return fw.avg_pool2d(x, [3, 3], [1, 1], "SAME", data_format=child.data_format.actual)

      def max_pool2d(x):
        with fw.name_scope("pool"):
          return fw.max_pool2d(x, [3, 3], [1, 1], "SAME", data_format=child.data_format.actual)

      if avg_or_max == "avg":
        self.layers.append(avg_pool2d)
      elif avg_or_max == "max":
        self.layers.append(max_pool2d)
      else:
        raise ValueError(f"Unknown pool {avg_or_max}")

      if start_idx is not None:
        self.layers.append(lambda x: child.data_format.pool_branch(x, start_idx, count))


  class LossModel(LayeredModel):
    def __init__(self, train):
      self.layers = [
        lambda x: fw.sparse_softmax_cross_entropy_with_logits(
          logits=x,
          labels=train),
        fw.reduce_mean]


  class TrainModel(LayeredModel):
    def __init__(self, train):
      self.layers = [
        lambda x: fw.argmax(x, axis=1),
        fw.to_int32,
        lambda x: fw.equal(x, train),
        fw.to_int32,
        fw.reduce_sum]

  # override
  def _build_train(self, y):
    loss = MacroChild.LossModel(y)
    train_acc = MacroChild.TrainModel(y)
    print("-" * 80)
    print("Build train graph")

    global_step = fw.get_or_create_global_step()
    train_op, lr, grad_norm, optimizer = get_train_ops(
      global_step,
      self.learning_rate,
      clip_mode=self.clip_mode,
      l2_reg=self.l2_reg,
      num_train_batches=self.num_train_batches,
      optim_algo=self.optim_algo)
    return loss, train_acc, global_step, train_op, lr, grad_norm, optimizer


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
        self.layers.append(lambda x, y: fw.shuffle_batch([x, y], child.batch_size, child.seed)) # 1
        if shuffle:

          def _pre_process(x):
            return self.data_format.child_init_preprocess(fw.image.random_flip_left_right(
              fw.image.random_crop(
                fw.pad(x, [[4, 4], [4, 4], [0, 0]]),
                [32, 32, 3],
                seed=self.seed),
              seed=self.seed))

          self.layers.append(lambda x: fw.map_fn(_pre_process, x, back_prop=False)) # 2
        else:
          self.layers.append(lambda x: x) # 2

    def __call__(self, x, y):
      with fw.device('/cpu:0'):
        x = self.layers[0](x)
        x_valid_shuffle, y_valid_shuffle = self.layers[1](x, y)
        x_valid_shuffle = self.layers[2](x_valid_shuffle)
        return x_valid_shuffle, y_valid_shuffle


  class ValidationRL(LayeredModel):
    def __init__(self):
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
        valid_shuffle_acc = self.layers[2](valid_shuffle_preds, y_valid_shuffle)
        valid_shuffle_acc = self.layers[3](valid_shuffle_acc)
        return self.layers[4](valid_shuffle_acc)

  def connect_controller(self, controller_model):
    if self.fixed_arc is None:
      self.sample_arc = controller_model.sample_arc
    else:
      self.sample_arc = np.array([int(x) for x in self.fixed_arc.split(" ") if x])

    self.loss, self.train_acc, self.global_step, train_op, lr, grad_norm, optimizer = self._build_train(self.y_train)
    self.valid_preds, self.valid_acc = self._build_valid(self.y_valid) # unused?
    self.test_preds, self.test_acc = self._build_test(self.y_test) # unused?
    return train_op, lr, grad_norm, optimizer