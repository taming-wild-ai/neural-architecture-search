from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time

import numpy as np
from absl import flags
import src.framework as fw

from src.controller import Controller
from src.utils import get_train_ops, DEFINE_float, ClipMode, Optimizer, LayeredModel
from src.common_ops import stack_lstm
from src.cifar10.micro_child import MicroChild # for child_num_cells
from src.cifar10.macro_child import MacroChild # for child_num_branches

from tensorflow.python.training import moving_averages

DEFINE_float("controller_op_tanh_reduce", 1.0, "")

class MicroController(Controller):
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
               **kwargs):
    super(MicroController, self).__init__()
    FLAGS = flags.FLAGS
    print("-" * 80)
    print("Building ConvController")

    self.num_cells = FLAGS.child_num_cells
    self.num_branches = FLAGS.child_num_branches

    self.lstm_size = lstm_size
    self.lstm_num_layers = lstm_num_layers
    self.lstm_keep_prob = lstm_keep_prob
    self.op_tanh_reduce = FLAGS.controller_op_tanh_reduce
    self.lr_dec_start = lr_dec_start
    self.lr_dec_every = lr_dec_every
    self.lr_dec_rate = lr_dec_rate
    self.clip_mode = ClipMode.new(clip_mode, grad_bound)
    self.optim_algo = Optimizer.new(optim_algo, FLAGS.controller_sync_replicas, FLAGS.controller_num_aggregate, FLAGS.controller_num_replicas)

    self.name = name

    self._create_params()
    self.sampler1 = MicroController.Sampler(self.num_cells, self.temperature, self.tanh_constant, self.op_tanh_reduce, self.lstm_num_layers, self.lstm_size, self.g_emb, self.w_emb, self.w_lstm, self.w_soft, self.b_soft, self.w_attn_1, self.w_attn_2, self.v_attn)
    self.sampler2 = MicroController.Sampler(self.num_cells, self.temperature, self.tanh_constant, self.op_tanh_reduce, self.lstm_num_layers, self.lstm_size, self.g_emb, self.w_emb, self.w_lstm, self.w_soft, self.b_soft, self.w_attn_1, self.w_attn_2, self.v_attn)
    _, _ = self.generate_sample_arc()

  def generate_sample_arc(self):
      self.current_normal_arc, entropy_1, log_prob_1, last_c, last_h = self.sampler1(None, None)
      self.current_reduce_arc, entropy_2, log_prob_2, _, _ = self.sampler1(last_c, last_h)
      self.current_entropy = entropy_1 + entropy_2
      self.current_log_prob = log_prob_1 + log_prob_2
      return self.current_normal_arc, self.current_reduce_arc

  def trainable_variables(self):
    new_vars = self.w_lstm + [self.g_emb, self.w_emb, self.w_soft, self.w_attn_1, self.w_attn_2, self.v_attn]
    return new_vars

  def _create_params(self):
    with fw.name_scope(self.name) as scope:
      initializer = fw.random_uniform_initializer(minval=-0.1, maxval=0.1)
      with fw.name_scope("lstm") as scope:
        self.w_lstm = []
        for layer_id in range(self.lstm_num_layers):
          with fw.name_scope("layer_{}".format(layer_id)) as scope:
            self.w_lstm.append(fw.Variable(initializer(shape=[2 * self.lstm_size, 4 * self.lstm_size]), name="w", import_scope=scope, trainable=True))

      self.g_emb = fw.Variable(initializer(shape=[1, self.lstm_size]), name="g_emb", import_scope=scope, trainable=True)
      with fw.name_scope("emb") as scope:
        self.w_emb = fw.Variable(initializer(shape=[self.num_branches, self.lstm_size]), name="w", import_scope=scope, trainable=True)
      with fw.name_scope("softmax") as scope:
        self.w_soft = fw.Variable(initializer(shape=[self.lstm_size, self.num_branches]), name="w", import_scope=scope, trainable=True)
        self.b_soft = fw.Variable(
          fw.constant_initializer(
            np.array([10.0, 10.0] + [0] * (self.num_branches - 2),
            dtype=np.float32))([1, self.num_branches]), import_scope=scope, trainable=True)

        self.b_soft_no_learn = fw.constant(
          np.reshape(
            np.array(
              [0.25, 0.25] + [-0.25] * (self.num_branches - 2),
              dtype=np.float32),
            [1, self.num_branches]),
          dtype=fw.float32)

      with fw.name_scope("attention") as scope:
        self.w_attn_1 = fw.Variable(initializer([self.lstm_size, self.lstm_size]), "w_1", import_scope=scope, trainable=True)
        self.w_attn_2 = fw.Variable(initializer([self.lstm_size, self.lstm_size]), "w_2", import_scope=scope, trainable=True)
        self.v_attn = fw.Variable(initializer([self.lstm_size, 1]), "v", import_scope=scope, trainable=True)


  class Sampler(LayeredModel):
      def __init__(self, num_cells, temperature, tanh_constant, op_tanh_reduce, lstm_num_layers, lstm_size, g_emb, w_emb, w_lstm, w_soft, b_soft, w_attn_1, w_attn_2, v_attn, use_bias=False):
          self.num_cells = num_cells
          self.temperature = temperature
          self.tanh_constant = tanh_constant
          self.op_tanh_reduce = op_tanh_reduce
          self.lstm_num_layers = lstm_num_layers
          self.lstm_size = lstm_size
          self.g_emb = g_emb
          self.w_emb = w_emb
          self.w_lstm = w_lstm
          self.w_soft = w_soft
          self.b_soft = b_soft
          self.w_attn_1 = w_attn_1
          self.w_attn_2 = w_attn_2
          self.v_attn = v_attn
          self.use_bias = use_bias

      def __call__(self, prev_c, prev_h):
          anchors = fw.TensorArray(
              fw.float32,
              size=self.num_cells + 2,
              clear_after_read=False)
          anchors_w_1 = fw.TensorArray(
              fw.float32,
              size=self.num_cells + 2,
              clear_after_read=False)
          arc_seq = fw.TensorArray(fw.int32, size=self.num_cells * 4)
          if prev_c is None:
              assert prev_h is None, "prev_c and prev_h must both be None"
              prev_c = [fw.zeros([1, self.lstm_size], fw.float32)
                        for _ in range(self.lstm_num_layers)]
              prev_h = [fw.zeros([1, self.lstm_size], fw.float32)
                        for _ in range(self.lstm_num_layers)]
          inputs = self.g_emb
          for layer_id in range(2):
              next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
              prev_c, prev_h = next_c, next_h
              anchors = anchors.write(layer_id, fw.zeros_like(next_h[-1]))
              anchors_w_1 = anchors_w_1.write(
                  layer_id,
                  fw.matmul(next_h[-1], self.w_attn_1))

          def _condition(layer_id, *args):
              return fw.less(layer_id, self.num_cells + 2)

          def _body(layer_id, inputs, prev_c, prev_h, anchors, anchors_w_1, arc_seq, entropy, log_prob):
              indices = fw.range(0, layer_id, dtype=fw.int32)
              start_id = 4 * (layer_id - 2)
              prev_layers = []
              for i in range(2):  # index_1, index_2
                  next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                  prev_c, prev_h = next_c, next_h
                  logit = fw.reshape(
                      fw.matmul(
                          fw.tanh(
                              fw.reshape(
                                  anchors_w_1.gather(indices),
                                  [layer_id, self.lstm_size]) + fw.matmul(
                                      next_h[-1],
                                      self.w_attn_2)),
                          self.v_attn),
                      [1, layer_id])
                  if self.temperature is not None:
                      logit /= self.temperature
                  if self.tanh_constant is not None:
                      logit = self.tanh_constant * fw.tanh(logit)
                  index = fw.reshape(fw.to_int32(fw.multinomial(logit, 1)), [1])
                  arc_seq = arc_seq.write(start_id + 2 * i, index)
                  curr_log_prob = fw.sparse_softmax_cross_entropy_with_logits(
                      logits=logit, labels=index)
                  log_prob += curr_log_prob
                  curr_ent = fw.stop_gradient(fw.softmax_cross_entropy_with_logits(
                      logits=logit, labels=fw.softmax(logit)))
                  entropy += curr_ent
                  prev_layers.append(anchors.read(fw.reduce_sum(index)))
                  inputs = prev_layers[-1]
              for i in range(2):  # op_1, op_2
                  next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                  prev_c, prev_h = next_c, next_h
                  logit = fw.matmul(next_h[-1], self.w_soft) + self.b_soft
                  if self.temperature is not None:
                      logit /= self.temperature
                  if self.tanh_constant is not None:
                      op_tanh = self.tanh_constant / self.op_tanh_reduce
                      logit = op_tanh * fw.tanh(logit)
                  if self.use_bias:
                      logit += self.b_soft_no_learn
                  op_id = fw.reshape(fw.to_int32(fw.multinomial(logit, 1)), [1])
                  arc_seq = arc_seq.write(start_id + 2 * i + 1, op_id)
                  curr_log_prob = fw.sparse_softmax_cross_entropy_with_logits(
                        logits=logit, labels=op_id)
                  log_prob += curr_log_prob
                  curr_ent = fw.stop_gradient(fw.softmax_cross_entropy_with_logits(
                        logits=logit, labels=fw.softmax(logit)))
                  entropy += curr_ent
                  inputs = fw.embedding_lookup(self.w_emb, op_id)
              next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
              anchors = anchors.write(layer_id, next_h[-1])
              anchors_w_1 = anchors_w_1.write(layer_id, fw.matmul(next_h[-1], self.w_attn_1))
              inputs = self.g_emb
              return (layer_id + 1, inputs, next_c, next_h, anchors, anchors_w_1, arc_seq, entropy, log_prob)

          loop_vars = [
              fw.constant(2, dtype=fw.int32, name="layer_id"),
              inputs,
              prev_c,
              prev_h,
              anchors,
              anchors_w_1,
              arc_seq,
              fw.constant([0.0], dtype=fw.float32, name="entropy"),
              fw.constant([0.0], dtype=fw.float32, name="log_prob")]
          loop_outputs = fw.while_loop(_condition, _body, loop_vars,
                                 parallel_iterations=1)
          arc_seq = loop_outputs[-3].stack()
          arc_seq = fw.reshape(arc_seq, [-1])
          entropy = fw.reduce_sum(loop_outputs[-2])
          log_prob = fw.reduce_sum(loop_outputs[-1])
          last_c = loop_outputs[-7]
          last_h = loop_outputs[-6]
          return arc_seq, entropy, log_prob, last_c, last_h


  def build_trainer(self, child_model, vrl):
    self.skip_rate = fw.constant(0.0, dtype=fw.float32)

    def valid_acc(child_logits, labels):
        return (fw.to_float(vrl(child_logits, labels)) /
            fw.to_float(child_model.batch_size))

    self.valid_acc = valid_acc

    def reward(logits, y_valid_shuffle):
      retval = self.valid_acc(logits, y_valid_shuffle)
      if self.entropy_weight is not None:
        retval += self.entropy_weight * self.current_entropy
      return retval

    self.baseline = fw.Variable(0.0, dtype=fw.float32)

    def loss(child_logits, y_valid_shuffle):
      with fw.control_dependencies([
        self.baseline.assign_sub((1 - self.bl_dec) * (self.baseline - reward(child_logits, y_valid_shuffle)))]):
        self.reward = fw.identity(reward(child_logits, y_valid_shuffle))
      retval = self.current_log_prob * (self.reward - self.baseline)
      return retval

    self.loss = loss
    self.train_step = fw.Variable(0, dtype=fw.int64, name="train_step")
    print("-" * 80)
    for var in self.trainable_variables():
      print(var)

    train_op, lr, optimizer = get_train_ops(
      self.train_step,
      self.learning_rate,
      clip_mode=self.clip_mode,
      optim_algo=self.optim_algo)
    return train_op, lr, optimizer
