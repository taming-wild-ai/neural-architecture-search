import sys
import os
import time

import numpy as np
import src.framework as fw

from src.controller import Controller
from src.utils import get_train_ops
from src.common_ops import stack_lstm


class MacroController(Controller):
  def __init__(self,
               search_for="both",
               search_whole_channels=False,
               num_layers=4,
               num_branches=6,
               out_filters=48,
               lstm_size=32,
               lstm_num_layers=2,
               lstm_keep_prob=1.0,
               tanh_constant=None,
               temperature=None,
               lr_init=1e-3,
               lr_dec_start=0,
               lr_dec_every=100,
               lr_dec_rate=0.9,
               l2_reg=0,
               entropy_weight=None,
               clip_mode=None,
               grad_bound=None,
               use_critic=False,
               bl_dec=0.999,
               optim_algo="adam",
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               skip_target=0.8,
               skip_weight=0.5,
               name="controller",
               *args,
               **kwargs):

    print("-" * 80)
    print("Building ConvController")

    self.search_for = search_for
    self.search_whole_channels = search_whole_channels
    self.num_layers = num_layers
    self.num_branches = num_branches
    self.out_filters = out_filters

    self.lstm_size = lstm_size
    self.lstm_num_layers = lstm_num_layers
    self.lstm_keep_prob = lstm_keep_prob
    self.tanh_constant = tanh_constant
    self.temperature = temperature
    self.lr_init = lr_init
    self.lr_dec_start = lr_dec_start
    self.lr_dec_every = lr_dec_every
    self.lr_dec_rate = lr_dec_rate
    self.l2_reg = l2_reg
    self.entropy_weight = entropy_weight
    self.clip_mode = clip_mode
    self.grad_bound = grad_bound
    self.use_critic = use_critic
    self.bl_dec = bl_dec

    self.skip_target = skip_target
    self.skip_weight = skip_weight

    self.optim_algo = optim_algo
    self.sync_replicas = sync_replicas
    self.num_aggregate = num_aggregate
    self.num_replicas = num_replicas
    self.name = name

    self._create_params()
    self._build_sampler()

  def _create_params(self):
    initializer = fw.random_uniform_initializer(minval=-0.1, maxval=0.1)
    with fw.variable_scope(self.name, initializer=initializer):
      with fw.variable_scope("lstm"):
        self.w_lstm = []
        for layer_id in range(self.lstm_num_layers):
          with fw.variable_scope("layer_{}".format(layer_id)):
            w = fw.get_variable(
              "w", [2 * self.lstm_size, 4 * self.lstm_size])
            self.w_lstm.append(w)

      self.g_emb = fw.get_variable("g_emb", [1, self.lstm_size])
      if self.search_whole_channels:
        with fw.variable_scope("emb"):
          self.w_emb = fw.get_variable(
            "w", [self.num_branches, self.lstm_size])
        with fw.variable_scope("softmax"):
          self.w_soft = fw.get_variable(
            "w", [self.lstm_size, self.num_branches])
      else:
        self.w_emb = {"start": [], "count": []}
        with fw.variable_scope("emb"):
          for branch_id in range(self.num_branches):
            with fw.variable_scope("branch_{}".format(branch_id)):
              self.w_emb["start"].append(fw.get_variable(
                "w_start", [self.out_filters, self.lstm_size]));
              self.w_emb["count"].append(fw.get_variable(
                "w_count", [self.out_filters - 1, self.lstm_size]));

        self.w_soft = {"start": [], "count": []}
        with fw.variable_scope("softmax"):
          for branch_id in range(self.num_branches):
            with fw.variable_scope("branch_{}".format(branch_id)):
              self.w_soft["start"].append(fw.get_variable(
                "w_start", [self.lstm_size, self.out_filters]));
              self.w_soft["count"].append(fw.get_variable(
                "w_count", [self.lstm_size, self.out_filters - 1]));

      with fw.variable_scope("attention"):
        self.w_attn_1 = fw.get_variable("w_1", [self.lstm_size, self.lstm_size])
        self.w_attn_2 = fw.get_variable("w_2", [self.lstm_size, self.lstm_size])
        self.v_attn = fw.get_variable("v", [self.lstm_size, 1])

  def _build_sampler(self):
    """Build the sampler ops and the log_prob ops."""

    print("-" * 80)
    print("Build controller sampler")
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
        logit = fw.matmul(next_h[-1], self.w_soft)
        if self.temperature is not None:
          logit /= self.temperature
        if self.tanh_constant is not None:
          logit = self.tanh_constant * fw.tanh(logit)
        if self.search_for == "macro" or self.search_for == "branch":
          branch_id = fw.multinomial(logit, 1)
          branch_id = fw.to_int32(branch_id)
          branch_id = fw.reshape(branch_id, [1])
        elif self.search_for == "connection":
          branch_id = fw.constant([0], dtype=fw.int32)
        else:
          raise ValueError("Unknown search_for {}".format(self.search_for))
        arc_seq.append(branch_id)
        log_prob = fw.sparse_softmax_cross_entropy_with_logits(
          logits=logit, labels=branch_id)
        log_probs.append(log_prob)
        entropy = fw.stop_gradient(log_prob * fw.exp(-log_prob))
        entropys.append(entropy)
        inputs = fw.embedding_lookup(self.w_emb, branch_id)
      else:
        for branch_id in range(self.num_branches):
          next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
          prev_c, prev_h = next_c, next_h
          logit = fw.matmul(next_h[-1], self.w_soft["start"][branch_id])
          if self.temperature is not None:
            logit /= self.temperature
          if self.tanh_constant is not None:
            logit = self.tanh_constant * fw.tanh(logit)
          start = fw.multinomial(logit, 1)
          start = fw.to_int32(start)
          start = fw.reshape(start, [1])
          arc_seq.append(start)
          log_prob = fw.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=start)
          log_probs.append(log_prob)
          entropy = fw.stop_gradient(log_prob * fw.exp(-log_prob))
          entropys.append(entropy)
          inputs = fw.embedding_lookup(self.w_emb["start"][branch_id], start)

          next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
          prev_c, prev_h = next_c, next_h
          logit = fw.matmul(next_h[-1], self.w_soft["count"][branch_id])
          if self.temperature is not None:
            logit /= self.temperature
          if self.tanh_constant is not None:
            logit = self.tanh_constant * fw.tanh(logit)
          mask = fw.range(0, limit=self.out_filters-1, delta=1, dtype=fw.int32)
          mask = fw.reshape(mask, [1, self.out_filters - 1])
          mask = fw.less_equal(mask, self.out_filters-1 - start)
          logit = fw.where(mask, x=logit, y=fw.fill(fw.shape(logit), -np.inf))
          count = fw.multinomial(logit, 1)
          count = fw.to_int32(count)
          count = fw.reshape(count, [1])
          arc_seq.append(count + 1)
          log_prob = fw.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=count)
          log_probs.append(log_prob)
          entropy = fw.stop_gradient(log_prob * fw.exp(-log_prob))
          entropys.append(entropy)
          inputs = fw.embedding_lookup(self.w_emb["count"][branch_id], count)

      next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
      prev_c, prev_h = next_c, next_h

      if layer_id > 0:
        query = fw.concat(anchors_w_1, axis=0)
        query = fw.tanh(query + fw.matmul(next_h[-1], self.w_attn_2))
        query = fw.matmul(query, self.v_attn)
        logit = fw.concat([-query, query], axis=1)
        if self.temperature is not None:
          logit /= self.temperature
        if self.tanh_constant is not None:
          logit = self.tanh_constant * fw.tanh(logit)

        skip = fw.multinomial(logit, 1)
        skip = fw.to_int32(skip)
        skip = fw.reshape(skip, [layer_id])
        arc_seq.append(skip)

        skip_prob = fw.sigmoid(logit)
        kl = skip_prob * fw.log(skip_prob / skip_targets)
        kl = fw.reduce_sum(kl)
        skip_penaltys.append(kl)

        log_prob = fw.sparse_softmax_cross_entropy_with_logits(
          logits=logit, labels=skip)
        log_probs.append(fw.reduce_sum(log_prob, keepdims=True))

        entropy = fw.stop_gradient(
          fw.reduce_sum(log_prob * fw.exp(-log_prob), keepdims=True))
        entropys.append(entropy)

        skip = fw.to_float(skip)
        skip = fw.reshape(skip, [1, layer_id])
        skip_count.append(fw.reduce_sum(skip))
        inputs = fw.matmul(skip, fw.concat(anchors, axis=0))
        inputs /= (1.0 + fw.reduce_sum(skip))
      else:
        inputs = self.g_emb

      anchors.append(next_h[-1])
      anchors_w_1.append(fw.matmul(next_h[-1], self.w_attn_1))

    arc_seq = fw.concat(arc_seq, axis=0)
    self.sample_arc = fw.reshape(arc_seq, [-1])

    entropys = fw.stack(entropys)
    self.sample_entropy = fw.reduce_sum(entropys)

    log_probs = fw.stack(log_probs)
    self.sample_log_prob = fw.reduce_sum(log_probs)

    skip_count = fw.stack(skip_count)
    self.skip_count = fw.reduce_sum(skip_count)

    skip_penaltys = fw.stack(skip_penaltys)
    self.skip_penaltys = fw.reduce_mean(skip_penaltys)

  def build_trainer(self, child_model):
    child_model.build_valid_rl()
    self.valid_acc = (fw.to_float(child_model.valid_shuffle_acc) /
                      fw.to_float(child_model.batch_size))
    self.reward = self.valid_acc

    normalize = fw.to_float(self.num_layers * (self.num_layers - 1) / 2)
    self.skip_rate = fw.to_float(self.skip_count) / normalize

    if self.entropy_weight is not None:
      self.reward += self.entropy_weight * self.sample_entropy

    self.sample_log_prob = fw.reduce_sum(self.sample_log_prob)
    self.baseline = fw.Variable(0.0, dtype=fw.float32)
    print(f"self.baseline = {self.baseline}, self.bl_dec = {self.bl_dec}, self.reward = {self.reward}")
    baseline_update = fw.assign_sub(
      self.baseline, (1 - self.bl_dec) * (self.baseline - self.reward))

    with fw.control_dependencies([baseline_update]):
      self.reward = fw.identity(self.reward)

    self.loss = self.sample_log_prob * (self.reward - self.baseline)
    if self.skip_weight is not None:
      self.loss += self.skip_weight * self.skip_penaltys

    self.train_step = fw.Variable(0, dtype=fw.int32, name="train_step")
    tf_variables = [var
        for var in fw.trainable_variables() if var.name.startswith(self.name)]
    print("-" * 80)
    for var in tf_variables:
      print(var)

    self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
      self.loss,
      tf_variables,
      self.train_step,
      clip_mode=self.clip_mode,
      grad_bound=self.grad_bound,
      l2_reg=self.l2_reg,
      lr_init=self.lr_init,
      lr_dec_start=self.lr_dec_start,
      lr_dec_every=self.lr_dec_every,
      lr_dec_rate=self.lr_dec_rate,
      optim_algo=self.optim_algo,
      sync_replicas=self.sync_replicas,
      num_aggregate=self.num_aggregate,
      num_replicas=self.num_replicas)

