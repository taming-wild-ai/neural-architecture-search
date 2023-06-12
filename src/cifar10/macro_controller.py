import sys
import os
import time

import numpy as np
from absl import flags
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
    FLAGS = flags.FLAGS
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
    self.sampler_logit = MacroController.SamplerLogit(self.num_layers, self.num_branches, self.out_filters, self.temperature, self.tanh_constant, self.lstm_size, self.lstm_num_layers, self.search_for, self.search_whole_channels, self.w_lstm, self.w_soft, self.w_emb, self.w_attn_1, self.w_attn_2, self.v_attn, self.g_emb)
    self.sample_arc = MacroController.SampleArc(self.num_layers, self.num_branches, self.search_whole_channels)
    self.sample_log_prob = MacroController.LogProbabilities(self.num_layers, self.num_branches, self.search_whole_channels)
    self.sample_entropy = MacroController.Entropy(self.num_layers, self.num_branches, self.search_whole_channels)
    _ = self.generate_sample_arc()
    self.skip_count = MacroController.SkipCount(self.num_layers, self.num_branches, self.search_whole_channels)
    self.skip_penaltys = MacroController.SkipPenalty(self.num_layers, self.num_branches, self.search_whole_channels, self.skip_target)

  def generate_sample_arc(self):
      self.current_controller_logits, self.current_branch_ids = self.sampler_logit()
      self.current_sample_arc = self.sample_arc(self.current_branch_ids)
      self.current_log_prob, self.current_log_prob_list = self.sample_log_prob(self.current_controller_logits, self.current_branch_ids)
      self.current_entropy = self.sample_entropy(self.current_log_prob_list)
      return self.current_sample_arc

  def trainable_variables(self):
    new_vars = self.w_lstm + [self.g_emb]
    if self.search_whole_channels:
      new_vars += [self.w_emb, self.w_soft]
    else:
      new_vars += self.w_emb['start'] + self.w_emb['count'] + self.w_soft['start'] + self.w_soft['count'] + [self.w_attn_1, self.w_attn_2, self.v_attn]
    return new_vars

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


  class SamplerLogit(LayeredModel):
      def __init__(self, num_layers, num_branches, out_filters, temperature, tanh_constant, lstm_size, lstm_num_layers, search_for, search_whole_channels, w_lstm, w_soft, w_emb, w_attn_1, w_attn_2, v_attn, g_emb):
          print("-" * 80)
          print("Build controller sampler")
          self.num_layers = num_layers
          self.num_branches = num_branches
          self.out_filters = out_filters
          self.search_whole_channels = search_whole_channels
          self.w_lstm = w_lstm
          self.w_soft = w_soft
          self.la = MacroController.LogitAdjuster(temperature, tanh_constant)
          self.bselect = MacroController.BranchSelector(search_for)
          self.w_emb = w_emb
          self.w_attn_1 = w_attn_1
          self.w_attn_2 = w_attn_2
          self.v_attn = v_attn
          self.g_emb = g_emb
          self.lstm_size = lstm_size
          self.lstm_num_layers = lstm_num_layers

      def __call__(self):
          """
          Return a tuple of logits and branch IDs, generated from the LSTM,
          suitable for passing to LogProbabilities.
          """
          anchors = []
          anchors_w_1 = []
          prev_c = [fw.zeros([1, self.lstm_size], fw.float32) for _ in
                    range(self.lstm_num_layers)]
          prev_h = [fw.zeros([1, self.lstm_size], fw.float32) for _ in
                    range(self.lstm_num_layers)]
          inputs = self.g_emb
          logits = []
          branch_ids = []
          for layer_id in range(self.num_layers):
              if self.search_whole_channels:
                  next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                  self.prev_c, self.prev_h = next_c, next_h
                  logit = self.la(fw.matmul(next_h[-1], self.w_soft))
                  branch_id = self.bselect(logit)
                  logits.append(logit)
                  branch_ids.append(branch_id)
                  inputs = fw.embedding_lookup(self.w_emb, branch_id)
              else:
                  layer_branch_ids = []
                  layer_logits = []
                  for branch_id in range(self.num_branches):
                      next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                      prev_c, prev_h = next_c, next_h
                      logit1 = self.la(fw.matmul(next_h[-1], self.w_soft['start'][branch_id]))
                      branch_id1 = fw.reshape(fw.to_int32(fw.multinomial(logit1, 1)), [1])
                      inputs = fw.embedding_lookup(self.w_emb['start'][branch_id], branch_id1)
                      next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                      prev_c, prev_h = next_c, next_h
                      logit2 = self.la(fw.matmul(next_h[-1], self.w_soft['count'][branch_id]))
                      logit2 = fw.where(
                          fw.less_equal(
                              fw.reshape(
                                  fw.range(0, limit=self.out_filters-1, delta=1, dtype=fw.int32),
                                  [1, self.out_filters - 1]),
                              self.out_filters-1 - branch_id1),
                              x=logit2,
                              y=fw.fill(fw.shape(logit2), -np.inf))
                      layer_logits.append([logit1, logit2])
                      branch_id2 = fw.reshape(fw.to_int32(fw.multinomial(logit2, 1)), [1])
                      layer_branch_ids.append([branch_id1, branch_id2])
                      self.inputs = fw.embedding_lookup(self.w_emb['count'][branch_id], branch_id2)
                  branch_ids.append(layer_branch_ids)
                  logits.append(layer_logits)
              next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
              prev_c, prev_h = next_c, next_h
              if layer_id > 0:
                  query = fw.matmul(fw.tanh(fw.concat(anchors_w_1, axis=0) + fw.matmul(next_h[-1], self.w_attn_2)), self.v_attn)
                  logit = self.la(fw.concat([-query, query], axis=1))
                  logits.append(logit)
                  skip = fw.reshape(fw.to_int32(fw.multinomial(logit, 1)), [layer_id])
                  branch_ids.append(skip)
              else:
                  inputs = self.g_emb
              anchors.append(next_h[-1])
              anchors_w_1.append(fw.matmul(next_h[-1], self.w_attn_1))
          return logits, branch_ids


  class SampleArc(LayeredModel):
      def __init__(self, num_layers, num_branches, search_whole_channels):
          self.num_layers = num_layers
          self.num_branches = num_branches
          self.search_whole_channels = search_whole_channels

      def __call__(self, branch_ids):
          """Generate the sample arc, based on branch IDs from SamplerLogit."""
          branch_ids_index = 0
          arc_seq = []
          for layer_id in range(self.num_layers):
              if self.search_whole_channels:
                  arc_seq.append(branch_ids[branch_ids_index])
              else:
                  for branch_id in range(self.num_branches):
                      arc_seq.append(branch_ids[branch_ids_index][branch_id][0])
                      arc_seq.append(branch_ids[branch_ids_index][branch_id][1] + 1)
              branch_ids_index += 1
              if layer_id > 0:
                  arc_seq.append(branch_ids[branch_ids_index])
                  branch_ids_index += 1
          retval = fw.concat(arc_seq, axis=0)
          retval = fw.reshape(retval, [-1])
          return retval


  class Entropy(LayeredModel):
      def __init__(self, num_layers, num_branches, search_whole_channels):
          self.num_layers = num_layers
          self.num_branches = num_branches
          self.search_whole_channels = search_whole_channels

      def __call__(self, log_probs):
          log_probs_index = 0
          entropys = []
          for layer_id in range(self.num_layers):
              if self.search_whole_channels:
                  entropys.append(fw.stop_gradient(log_probs[log_probs_index] * fw.exp(-log_probs[layer_id])))
              else:
                  for branch_id in range(self.num_branches):
                      entropys.append(fw.stop_gradient(log_probs[log_probs_index][branch_id][0] * fw.exp(-log_probs[log_probs_index][branch_id][0])))
                      entropys.append(fw.stop_gradient(log_probs[log_probs_index][branch_id][1] * fw.exp(-log_probs[log_probs_index][branch_id][1])))
              log_probs_index += 1
              if layer_id > 0:
                  entropys.append(fw.stop_gradient(
                      fw.reduce_sum(log_probs[log_probs_index] * fw.exp(-log_probs[log_probs_index]), keepdims=True)))
                  log_probs_index += 1
          retval = fw.stack(entropys)
          retval = fw.reduce_sum(entropys)
          return retval


  class LogProbabilities(LayeredModel):
      def __init__(self, num_layers, num_branches, search_whole_channels):
          self.num_layers = num_layers
          self.num_branches = num_branches
          self.search_whole_channels = search_whole_channels

      def __call__(self, logits, branch_ids):
          sequence_index = 0
          log_probs = []
          stackable_log_probs = []
          for layer_id in range(self.num_layers):
              if self.search_whole_channels:
                  log_prob = fw.sparse_softmax_cross_entropy_with_logits(
                      logits=logits[sequence_index],
                      labels=branch_ids[sequence_index])
                  log_probs.append(log_prob)
                  stackable_log_probs.append(log_prob)
              else:
                  layer_log_probs = []
                  for branch_id in range(self.num_branches):
                      log_prob1 = fw.sparse_softmax_cross_entropy_with_logits(
                          logits=logits[sequence_index][branch_id][0],
                          labels=branch_ids[sequence_index][branch_id][0])
                      stackable_log_probs.append(log_prob1)
                      log_prob2 = fw.sparse_softmax_cross_entropy_with_logits(
                          logits=logits[sequence_index][branch_id][1],
                          labels=branch_ids[sequence_index][branch_id][1])
                      layer_log_probs.append([log_prob1, log_prob2])
                      stackable_log_probs.append(log_prob2)
                  log_probs.append(layer_log_probs)
              sequence_index += 1
              if layer_id > 0:
                  log_prob = fw.reduce_sum(
                      fw.sparse_softmax_cross_entropy_with_logits(
                          logits=logits[sequence_index], labels=branch_ids[sequence_index]),
                      keepdims=True)
                  log_probs.append(log_prob)
                  stackable_log_probs.append(log_prob)
                  sequence_index += 1
          log_prob = fw.stack(stackable_log_probs)
          log_prob = fw.reduce_sum(log_prob)
          retval = (log_prob, log_probs)
          return retval


  class SkipCount(LayeredModel):
      def __init__(self, num_layers, num_branches, search_whole_channels):
          self.num_layers = num_layers
          self.num_branches = num_branches
          self.search_whole_channels = search_whole_channels

      def __call__(self, branch_ids):
          branch_ids_index = 0
          skip_count = []
          for layer_id in range(self.num_layers):
              branch_ids_index += 1
              if layer_id > 0:
                  skip = fw.reshape(fw.to_float(branch_ids[branch_ids_index]), [1, layer_id])
                  skip_count.append(fw.reduce_sum(skip))
                  branch_ids_index += 1
          retval = fw.stack(skip_count)
          retval = fw.reduce_sum(retval)
          return retval


  class SkipPenalty(LayeredModel):
      def __init__(self, num_layers, num_branches, search_whole_channels, skip_target):
          self.num_layers = num_layers
          self.num_branches = num_branches
          self.search_whole_channels = search_whole_channels
          self.skip_target = skip_target

      def __call__(self, logits):
          logits_index = 0
          skip_penaltys = []
          skip_targets = fw.constant([1.0 - self.skip_target, self.skip_target],
                                    dtype=fw.float32)
          for layer_id in range(self.num_layers):
              logits_index += 1
              if layer_id > 0:
                  skip_prob = fw.sigmoid(logits[logits_index])
                  skip_penaltys.append(fw.reduce_sum(skip_prob * fw.log(skip_prob / skip_targets)))
                  logits_index += 1
          retval = fw.stack(skip_penaltys)
          retval = fw.reduce_sum(retval)
          return retval


  def build_trainer(self, child_model, vrl):
    self.skip_rate = lambda branch_ids: fw.to_float(self.skip_count(branch_ids)) / fw.to_float(self.num_layers * (self.num_layers - 1) / 2)

    def valid_acc(child_logits, y_valid_shuffle):
        retval = (
            fw.to_float(vrl(child_logits, y_valid_shuffle)) /
            fw.to_float(child_model.batch_size))
        return retval

    self.valid_acc = valid_acc

    def reward(logits, y_valid_shuffle, log_probs):
      retval = self.valid_acc(logits, y_valid_shuffle)
      if self.entropy_weight is not None:
        retval += self.entropy_weight * self.sample_entropy(log_probs)
      return retval

    self.baseline = fw.Variable(0.0, dtype=fw.float32)

    def loss(child_logits, y_valid_shuffle):
        with fw.control_dependencies([
            self.baseline.assign_sub((1 - self.bl_dec) * (self.baseline - reward(child_logits, y_valid_shuffle, self.current_log_prob_list)))]):
            self.reward = fw.identity(reward(child_logits, y_valid_shuffle, self.current_log_prob_list))
        retval = self.current_log_prob * (self.reward - self.baseline)
        if self.skip_weight is not None:
            retval += self.skip_weight * self.skip_penaltys(self.current_controller_logits)
        return retval

    self.loss = loss
    self.train_step = fw.Variable(0, dtype=fw.int32, name="train_step")
    print("-" * 80)
    for var in self.trainable_variables():
      print(var)

    train_op, lr, optimizer = get_train_ops(
      self.train_step,
      self.learning_rate,
      clip_mode=self.clip_mode,
      l2_reg=self.l2_reg,
      optim_algo=self.optim_algo)
    return train_op, lr, optimizer
