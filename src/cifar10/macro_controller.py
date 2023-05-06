import sys
import os
import time

import numpy as np
import src.framework as fw

from src.controller import Controller
from src.utils import get_train_ops, Optimizer, ClipMode
from src.common_ops import stack_lstm

from src.utils import DEFINE_boolean, DEFINE_float, ClipMode, LayeredModel

DEFINE_boolean("controller_search_whole_channels", False, "")
DEFINE_float("controller_skip_target", 0.8, "")
DEFINE_float("controller_skip_weight", 0.0, "")

class MacroController(Controller):
  def __init__(self,
               lstm_size=32,
               lstm_num_layers=2,
               lstm_keep_prob=1.0,
               lr_dec_start=0,
               lr_dec_every=100,
               lr_dec_rate=0.9,
               clip_mode=None,
               grad_bound=None,
               optim_algo="adam",
               name="controller",
               *args,
               **kwargs):
    super(MacroController, self).__init__()
    FLAGS = fw.FLAGS
    print("-" * 80)
    print("Building ConvController")

    self.search_whole_channels = FLAGS.controller_search_whole_channels
    self.num_layers = FLAGS.child_num_layers # 4
    self.num_branches = FLAGS.child_num_branches # 6
    self.out_filters = FLAGS.child_out_filters # 48

    self.lstm_size = lstm_size
    self.lstm_num_layers = lstm_num_layers
    self.lstm_keep_prob = lstm_keep_prob
    self.lr_dec_start = lr_dec_start
    self.lr_dec_every = lr_dec_every
    self.lr_dec_rate = lr_dec_rate
    self.clip_mode = ClipMode.new(clip_mode, grad_bound)

    self.skip_target = FLAGS.controller_skip_target
    self.skip_weight = FLAGS.controller_skip_weight

    self.optim_algo = Optimizer.new(optim_algo, FLAGS.controller_sync_replicas, FLAGS.controller_num_aggregate, FLAGS.controller_num_replicas)
    self.name = name

    self._create_params()

    self.sample_arc, self.sample_entropy, self.sample_log_prob, self.skip_count, self.skip_penaltys = self._build_sampler()

  def _create_params(self):
    initializer = fw.random_uniform_initializer(minval=-0.1, maxval=0.1)
    with fw.name_scope(self.name):
      with fw.name_scope("lstm"):
        self.w_lstm = []
        for layer_id in range(self.lstm_num_layers):
          with fw.name_scope("layer_{}".format(layer_id)):
            self.w_lstm.append(
              fw.Variable(
                initializer([2 * self.lstm_size, 4 * self.lstm_size]),
                name="w",
                trainable=True))
      self.g_emb = fw.Variable(initializer([1, self.lstm_size]), name="g_emb", trainable=True)
      if self.search_whole_channels:
        with fw.name_scope("emb"):
          self.w_emb = fw.Variable(
            initializer([self.num_branches, self.lstm_size]),
            name="w",
            trainable=True)
        with fw.name_scope("softmax"):
          self.w_soft = fw.Variable(
            initializer([self.lstm_size, self.num_branches]),
            name="w",
            trainable=True)
      else:
        self.w_emb = {"start": [], "count": []}
        with fw.name_scope("emb"):
          for branch_id in range(self.num_branches):
            with fw.name_scope("branch_{}".format(branch_id)):
              self.w_emb["start"].append(
                fw.Variable(
                  initializer([self.out_filters, self.lstm_size]),
                  name="w_start",
                  trainable=True));
              self.w_emb["count"].append(
                fw.Variable(
                  initializer([self.out_filters - 1, self.lstm_size]),
                  name="w_count",
                  trainable=True));

        self.w_soft = {"start": [], "count": []}
        with fw.name_scope("softmax"):
          for branch_id in range(self.num_branches):
            with fw.name_scope("branch_{}".format(branch_id)):
              self.w_soft["start"].append(
                fw.Variable(
                  initializer([self.lstm_size, self.out_filters]),
                  name="w_start",
                  trainable=True));
              self.w_soft["count"].append(
                fw.Variable(
                  initializer([self.lstm_size, self.out_filters - 1]),
                  name="w_count",
                  trainable=True));

      with fw.name_scope("attention"):
        self.w_attn_1 = fw.Variable(initializer([self.lstm_size, self.lstm_size]), name="w_1", trainable=True)
        self.w_attn_2 = fw.Variable(initializer([self.lstm_size, self.lstm_size]), name="w_2", trainable=True)
        self.v_attn = fw.Variable(initializer([self.lstm_size, 1]), name="v", trainable=True)


  class LogitAdjuster(LayeredModel):
    def __init__(self, temperature, tanh_constant):
      self.layers = []
      if temperature is not None:
        self.layers.append(lambda logit: logit / temperature)
      if tanh_constant is not None:
        self.layers.append(lambda logit: tanh_constant * fw.tanh(logit))


  class BranchSelector(LayeredModel):
    def __init__(self, search_for):
      if search_for == "macro" or search_for == "branch":
        self.layers = [
          lambda logit: fw.multinomial(logit, 1),
          fw.to_int32,
          lambda logit: fw.reshape(logit, [1])]
      elif self.search_for == "connection":
        self.layers = [
          lambda _: fw.constant([0], dtype=fw.int32)]
      else:
        raise ValueError("Unknown search_for {}".format(self.search_for))


  def _build_sampler(self):
    """Build the sampler ops and the log_prob ops."""

    print("-" * 80)
    print("Build controller sampler")
    la = MacroController.LogitAdjuster(self.temperature, self.tanh_constant)
    bselect = MacroController.BranchSelector(self.search_for)
    anchors = []
    anchors_w_1 = []

    arc_seq = []
    entropys = []
    log_probs = []
    skip_count = []
    skip_penaltys = []

    prev_c = [fw.zeros([1, self.lstm_size], fw.float32) for _ in
              range(self.lstm_num_layers)]
    prev_h = [fw.zeros([1, self.lstm_size], fw.float32) for _ in
              range(self.lstm_num_layers)]
    inputs = self.g_emb
    skip_targets = fw.constant([1.0 - self.skip_target, self.skip_target],
                               dtype=fw.float32)
    for layer_id in range(self.num_layers):
      if self.search_whole_channels:
        next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
        prev_c, prev_h = next_c, next_h
        logit = la(fw.matmul(next_h[-1], self.w_soft))
        branch_id = bselect(logit)
        arc_seq.append(branch_id)
        log_prob = fw.sparse_softmax_cross_entropy_with_logits(
          logits=logit,
          labels=branch_id)
        log_probs.append(log_prob)
        entropys.append(fw.stop_gradient(log_prob * fw.exp(-log_prob)))
        inputs = fw.embedding_lookup(self.w_emb, branch_id)
      else:
        for branch_id in range(self.num_branches):
          next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
          prev_c, prev_h = next_c, next_h
          logit = la(fw.matmul(next_h[-1], self.w_soft["start"][branch_id]))
          start = fw.reshape(fw.to_int32(fw.multinomial(logit, 1)), [1])
          arc_seq.append(start)
          log_prob = fw.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=start)
          log_probs.append(log_prob)
          entropys.append(fw.stop_gradient(log_prob * fw.exp(-log_prob)))
          inputs = fw.embedding_lookup(self.w_emb["start"][branch_id], start)

          next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
          prev_c, prev_h = next_c, next_h
          logit = la(fw.matmul(next_h[-1], self.w_soft["count"][branch_id]))
          logit = fw.where(
            fw.less_equal(
              fw.reshape(
                fw.range(0, limit=self.out_filters-1, delta=1, dtype=fw.int32),
                [1, self.out_filters - 1]),
              self.out_filters-1 - start),
              x=logit,
              y=fw.fill(fw.shape(logit), -np.inf))
          count = fw.reshape(fw.to_int32(fw.multinomial(logit, 1)), [1])
          arc_seq.append(count + 1)
          log_prob = fw.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=count)
          log_probs.append(log_prob)
          entropys.append(fw.stop_gradient(log_prob * fw.exp(-log_prob)))
          inputs = fw.embedding_lookup(self.w_emb["count"][branch_id], count)

      next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
      prev_c, prev_h = next_c, next_h

      if layer_id > 0:
        query = fw.matmul(fw.tanh(fw.concat(anchors_w_1, axis=0) + fw.matmul(next_h[-1], self.w_attn_2)), self.v_attn)
        logit = la(fw.concat([-query, query], axis=1))

        skip = fw.reshape(fw.to_int32(fw.multinomial(logit, 1)), [layer_id])
        arc_seq.append(skip)

        skip_prob = fw.sigmoid(logit)
        skip_penaltys.append(fw.reduce_sum(skip_prob * fw.log(skip_prob / skip_targets)))

        log_prob = fw.sparse_softmax_cross_entropy_with_logits(
          logits=logit, labels=skip)
        log_probs.append(fw.reduce_sum(log_prob, keepdims=True))

        entropys.append(
          fw.stop_gradient(
            fw.reduce_sum(log_prob * fw.exp(-log_prob), keepdims=True)))

        skip = fw.reshape(fw.to_float(skip), [1, layer_id])
        skip_count.append(fw.reduce_sum(skip))
        inputs = fw.matmul(skip, fw.concat(anchors, axis=0)) / (1.0 + fw.reduce_sum(skip))
      else:
        inputs = self.g_emb

      anchors.append(next_h[-1])
      anchors_w_1.append(fw.matmul(next_h[-1], self.w_attn_1))

    return(
      fw.reshape(fw.concat(arc_seq, axis=0), [-1]),
      fw.reduce_sum(fw.stack(entropys)),
      fw.reduce_sum(fw.stack(log_probs)),
      fw.reduce_sum(fw.stack(skip_count)),
      fw.reduce_mean(fw.stack(skip_penaltys)))

  def build_trainer(self, child_model):
    shuffle, vrl = child_model.build_valid_rl()
    x_valid_shuffle, y_valid_shuffle = shuffle(child_model.images['valid_original'], child_model.labels['valid_original'])
    model = child_model.Model(child_model, True, True)
    logits = model(x_valid_shuffle)
    valid_shuffle_acc = vrl(logits, y_valid_shuffle)
    self.valid_acc = (fw.to_float(valid_shuffle_acc) /
                      fw.to_float(child_model.batch_size))
    reward = self.valid_acc

    self.skip_rate = fw.to_float(self.skip_count) / fw.to_float(self.num_layers * (self.num_layers - 1) / 2)

    if self.entropy_weight is not None:
      reward += self.entropy_weight * self.sample_entropy

    self.sample_log_prob = fw.reduce_sum(self.sample_log_prob)
    self.baseline = fw.Variable(0.0, dtype=fw.float32)

    with fw.control_dependencies([
      self.baseline.assign_sub((1 - self.bl_dec) * (self.baseline - reward))]):
      self.reward = fw.identity(reward)

    self.loss = self.sample_log_prob * (self.reward - self.baseline)
    if self.skip_weight is not None:
      self.loss += self.skip_weight * self.skip_penaltys

    self.train_step = fw.Variable(0, dtype=fw.int32, name="train_step")
    tf_variables = [var
        for var in fw.trainable_variables() if var.name.startswith(self.name)]
    print("-" * 80)
    for var in tf_variables:
      print(var)

    train_op, lr, grad_norm, optimizer = get_train_ops(
      self.train_step,
      self.learning_rate,
      clip_mode=self.clip_mode,
      l2_reg=self.l2_reg,
      optim_algo=self.optim_algo)
    return train_op(self.loss, tf_variables), lr, grad_norm(self.loss, tf_variables), optimizer

