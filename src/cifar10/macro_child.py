from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import src.framework as fw

from src.cifar10.child import Child
from src.cifar10.image_ops import batch_norm
from src.cifar10.image_ops import batch_norm_with_mask

from src.utils import count_model_params
from src.utils import get_train_ops
from src.utils import DEFINE_integer

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
    return self.data_format.get_strides(stride)

  def _factorized_reduction(self, x, out_filters, stride, is_training):
    """Reduces the shape of x without information loss due to striding."""
    assert out_filters % 2 == 0, (
        "Need even number of filters when using this factorized reduction.")
    if stride == 1:
      with fw.variable_scope("path_conv"):
        return batch_norm(
          fw.conv2d(
            x,
            fw.create_weight("w", [1, 1, self.data_format.get_C(x), out_filters]),
            [1, 1, 1, 1],
            "SAME",
            data_format=self.data_format.name),
          is_training,
          self.data_format)

    stride_spec = self._get_strides(stride)
    # Skip path 1
    path1 = fw.avg_pool(
        x, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format.name)
    with fw.variable_scope("path1_conv"):
      path1 = fw.conv2d(
        path1,
        fw.create_weight("w", [1, 1, self.data_format.get_C(path1), out_filters // 2]),
        [1, 1, 1, 1],
        "SAME",
        data_format=self.data_format.name)

    # Skip path 2
    # First pad with 0"s on the right and bottom, then shift the filter to
    # include those 0"s that were added.
    path2, concat_axis = self.data_format.factorized_reduction(x)

    path2 = fw.avg_pool(
        path2, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format.name)
    with fw.variable_scope("path2_conv"):
      path2 = fw.conv2d(
        path2,
        fw.create_weight("w", [1, 1, self.data_format.get_C(path2), out_filters // 2]),
        [1, 1, 1, 1],
        "SAME",
        data_format=self.data_format.name)

    # Concat and apply BN
    return batch_norm(
      fw.concat(values=[path1, path2], axis=concat_axis),
      is_training,
      self.data_format)

  def _model(self, images, is_training, reuse=False):
    with fw.variable_scope(self.name, reuse=reuse):
      layers = []

      out_filters = self.out_filters
      with fw.variable_scope("stem_conv"):
        layers.append(
          batch_norm(
            fw.conv2d(
              images,
              fw.create_weight("w", [3, 3, 3, out_filters]),
              [1, 1, 1, 1],
              "SAME",
              data_format=self.data_format.name),
            is_training,
            self.data_format))

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

      x = self.data_format.global_avg_pool(x)
      if is_training:
        x = fw.dropout(x, self.keep_prob)
      with fw.variable_scope("fc"):
        x = fw.matmul(x, fw.create_weight("w", [self.data_format.get_C(x), 10]))
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
      inp_h, inp_w = self.data_format.get_HW(inputs)

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

      self.data_format.set_shape(out, inp_h, inp_w, out_filters)
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
        w_mask = fw.constant([False] * (self.num_branches * out_filters), fw.bool)
        new_range = fw.range(0, self.num_branches * out_filters, dtype=fw.int32)
        for i in range(self.num_branches):
          start = out_filters * i + count[2 * i]
          w_mask = fw.logical_or(w_mask, fw.logical_and(
            start <= new_range, new_range < start + count[2 * i + 1]))
        w = fw.reshape(
          fw.boolean_mask(
            fw.create_weight("w", [self.num_branches * out_filters, out_filters]),
            w_mask),
            [1, 1, -1,
            out_filters])

        inp = prev_layers[-1]
        branches = self.data_format.enas_layer(inp, branches)
        out = fw.relu(
          batch_norm(
            fw.conv2d(
              branches,
              w,
              [1, 1, 1, 1],
              "SAME",
              data_format=self.data_format.name),
            is_training,
            self.data_format))

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
        out = batch_norm(fw.add_n(res_layers), is_training, self.data_format)

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
      inp_c = self.data_format.get_C(inputs)

      count = self.sample_arc[start_idx]
      if count in [0, 1, 2, 3]:
        filter_size = [3, 3, 5, 5][count]
        with fw.variable_scope("conv_1x1"):
          out = batch_norm(
            fw.conv2d(
              fw.relu(inputs),
              fw.create_weight("w", [1, 1, inp_c, out_filters]),
              [1, 1, 1, 1],
              "SAME",
              data_format=self.data_format.name),
            is_training,
            self.data_format)

        with fw.variable_scope("conv_{0}x{0}".format(filter_size)):
          out = batch_norm(
            fw.conv2d(
              fw.relu(out),
              fw.create_weight("w", [filter_size, filter_size, out_filters, out_filters]),
              [1, 1, 1, 1],
              "SAME",
              data_format=self.data_format.name),
            is_training,
            self.data_format)
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
        branches = self.data_format.fixed_layer(branches)
        out = batch_norm(
          fw.conv2d(
            fw.relu(branches),
            fw.create_weight("w", [1, 1, total_out_channels, out_filters]),
            [1, 1, 1, 1],
            "SAME",
            data_format=self.data_format.name),
          is_training,
          self.data_format)

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

      prev = self.data_format.fixed_layer(prev)

      with fw.variable_scope("skip"):
        out = batch_norm(
          fw.conv2d(
            fw.relu(prev),
            fw.create_weight("w", [1, 1, total_skip_channels * out_filters, out_filters]),
            [1, 1, 1, 1],
            "SAME",
            data_format=self.data_format.name),
          is_training,
          self.data_format)

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

    inp_c = self.data_format.get_C(inputs)

    with fw.variable_scope("inp_conv_1"):
      x = fw.relu(
        batch_norm(
          fw.conv2d(
            inputs,
            fw.create_weight("w", [1, 1, inp_c, out_filters]),
            [1, 1, 1, 1],
            "SAME",
            data_format=self.data_format.name),
          is_training,
          self.data_format))

    with fw.variable_scope("out_conv_{}".format(filter_size)):
      if start_idx is None:
        if separable:
          x = batch_norm(
            fw.separable_conv2d(
              x,
              fw.create_weight("w_depth", [self.filter_size, self.filter_size, out_filters, ch_mul]),
              fw.create_weight("w_point", [1, 1, out_filters * ch_mul, count]),
              strides=[1, 1, 1, 1],
              padding="SAME",
              data_format=self.data_format.name),
            is_training,
            self.data_format)
        else:
          x = batch_norm(
            fw.conv2d(
              x,
              fw.create_weight("w", [filter_size, filter_size, inp_c, count]),
              [1, 1, 1, 1],
              "SAME",
              data_format=self.data_format.name),
            is_training,
            self.data_format)
      else:
        mask = fw.range(0, out_filters, dtype=fw.int32)
        if separable:
          x = fw.separable_conv2d(
            x,
            fw.create_weight("w_depth", [filter_size, filter_size, out_filters, ch_mul]),
            fw.reshape(
              fw.transpose(
                fw.create_weight("w_point", [out_filters, out_filters * ch_mul])[start_idx:start_idx+count, :],
                [1, 0]),
              [1, 1, out_filters * ch_mul, count]),
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format=self.data_format.name)
        else:
          x = fw.conv2d(
            x,
            fw.transpose(
              fw.transpose(
                fw.create_weight("w", [filter_size, filter_size, out_filters, out_filters]),
                [3, 0, 1, 2])[start_idx:start_idx+count, :, :, :],
              [1, 2, 3, 0]),
            [1, 1, 1, 1],
            "SAME",
            data_format=self.data_format.name)
        x = batch_norm_with_mask(
          x,
          is_training,
          fw.logical_and(start_idx <= mask, mask < start_idx + count),
          out_filters,
          data_format=self.data_format.name)
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

    inp_c = self.data_format.get_C(inputs)

    with fw.variable_scope("conv_1"):
      x = fw.relu(
        batch_norm(
          fw.conv2d(
            inputs,
            fw.create_weight("w", [1, 1, inp_c, self.out_filters]),
            [1, 1, 1, 1],
            "SAME",
            data_format=self.data_format.name),
          is_training,
          self.data_format))

    with fw.variable_scope("pool"):
      if avg_or_max == "avg":
        x = fw.avg_pool2d(
          x, [3, 3], [1, 1], "SAME", data_format=self.data_format.actual)
      elif avg_or_max == "max":
        x = fw.max_pool2d(
          x, [3, 3], [1, 1], "SAME", data_format=self.data_format.actual)
      else:
        raise ValueError("Unknown pool {}".format(avg_or_max))

      if start_idx is not None:
        x = self.data_format.pool_branch(x, start_idx, count)

    return x

  # override
  def _build_train(self):
    print("-" * 80)
    print("Build train graph")
    logits = self._model(self.x_train, is_training=True)
    self.loss = fw.reduce_mean(
      fw.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=self.y_train))

    self.train_acc = fw.reduce_sum(
      fw.to_int32(fw.equal(fw.to_int32(fw.argmax(logits, axis=1)), self.y_train)))

    tf_variables = [var
        for var in fw.trainable_variables() if var.name.startswith(self.name)]
    self.num_vars = count_model_params(tf_variables)
    print("Model has {} params".format(self.num_vars))

    self.global_step = fw.get_or_create_global_step()
    self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
      self.loss,
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
      self.valid_preds = fw.to_int32(
        fw.argmax(
          self._model(self.x_valid, False, reuse=True),
          axis=1))
      self.valid_acc = fw.reduce_sum(
        fw.to_int32(
          fw.equal(self.valid_preds, self.y_valid)))

  # override
  def _build_test(self):
    print("-" * 80)
    print("Build test graph")
    self.test_preds = fw.to_int32(
      fw.argmax(self._model(self.x_test, False, reuse=True), axis=1))
    self.test_acc = fw.reduce_sum(fw.to_int32(fw.equal(self.test_preds, self.y_test)))

  # override
  def build_valid_rl(self, shuffle=False):
    print("-" * 80)
    print("Build valid graph on shuffled data")
    with fw.device("/gpu:0"):
      # shuffled valid data: for choosing validation model
      if not shuffle:
        self.images["valid_original"] = self.data_format.child_init(self.images["valid_original"])
      x_valid_shuffle, y_valid_shuffle = fw.shuffle_batch(
        [self.images["valid_original"], self.labels["valid_original"]],
        self.batch_size,
        self.seed)

      def _pre_process(x):
        return self.data_format.child_init_preprocess(fw.image.random_flip_left_right(
          fw.image.random_crop(
            fw.pad(x, [[4, 4], [4, 4], [0, 0]]),
            [32, 32, 3],
            seed=self.seed),
          seed=self.seed))

      if shuffle:
        x_valid_shuffle = fw.map_fn(
          _pre_process, x_valid_shuffle, back_prop=False)

    self.valid_shuffle_acc = fw.equal(
      fw.to_int32(
        fw.argmax(self._model(x_valid_shuffle, False, reuse=True), axis=1)),
        y_valid_shuffle)
    self.valid_shuffle_acc = fw.to_int32(self.valid_shuffle_acc)
    self.valid_shuffle_acc = fw.reduce_sum(self.valid_shuffle_acc)

  def connect_controller(self, controller_model):
    if self.fixed_arc is None:
      self.sample_arc = controller_model.sample_arc
    else:
      self.sample_arc = np.array([int(x) for x in self.fixed_arc.split(" ") if x])

    self._build_train()
    self._build_valid()
    self._build_test()

