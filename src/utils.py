from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
from absl import app
from absl import flags
from absl import logging

import src.framework as fw


class Optimizer(object):
  class Momentum:
    def __init__(self, sync, aggregates, replicas):
      if sync:
        assert aggregates is not None, "Need num_aggregate to sync."
        assert replicas is not None, "Need num_replicas to sync."
      self.sync = sync
      self.aggregates = aggregates
      self.replicas = replicas

    def get(self, lr, moving_average):
      opt = fw.Optimizer.Momentum(lr)
      if self.sync:
        opt = fw.Optimizer.SyncReplicas(opt, self.aggregates, self.replicas)
      if moving_average is not None:
        opt = fw.Optimizer.MovingAverage(opt, moving_average)
      return opt

  class SGD:
    def __init__(self, sync, aggregates, replicas):
      self.sync = sync
      self.aggregates = aggregates
      self.replicas = replicas

    def get(self, lr, moving_average):
      opt = fw.Optimizer.SGD(lr)
      if self.sync:
        opt = fw.Optimizer.SyncReplicas(opt, self.aggregates, self.replicas)
      if moving_average is not None:
        opt = fw.Optimizer.MovingAverage(opt, moving_average)
      return opt


  class Adam:
    def __init__(self, sync, aggregates, replicas):
      self.sync = sync
      self.aggregates = aggregates
      self.replicas = replicas

    def get(self, lr, moving_average):
      opt = fw.Optimizer.Adam(lr)
      if self.sync:
        opt = fw.Optimizer.SyncReplicas(opt, self.aggregates, self.replicas)
      if moving_average is not None:
        opt = fw.Optimizer.MovingAverage(opt, moving_average)
      return opt


  def __init__(self):
    raise AttributeError('Use factory method `new` instead.')

  @staticmethod
  def new(algo, sync_replicas, num_aggregate, num_replicas):
    return {
      "momentum": Optimizer.Momentum(sync_replicas, num_aggregate, num_replicas),
      "sgd": Optimizer.SGD(sync_replicas, num_aggregate, num_replicas),
      "adam": Optimizer.Adam(sync_replicas, num_aggregate, num_replicas)
    }[algo]


class ClipMode(object):
  class Null:
    def __init__(self, _):
      pass

    def clip(self, grads):
      return grads


  class Global:
    def __init__(self, bound):
      assert bound is not None, "Need grad_bound to clip gradients."
      self.bound = bound

    def clip(self, grads):
      grads, _ = fw.clip_by_global_norm(grads, self.bound)
      return grads


  class Norm:
    def __init__(self, bound):
      assert bound is not None, "Need grad_bound to clip gradients."
      self.bound = bound

    def clip(self, grads):
      clipped = []
      for g in grads:
        if isinstance(g, fw.IndexedSlices):
          c_g = fw.IndexedSlices(g.indices, fw.clip_by_norm(g.values, self.bound))
        else:
          c_g = fw.clip_by_norm(g, self.bound)
        clipped.append(c_g)
      return clipped


  def __init__(self):
    raise AttributeError("Use factory method `new` instead.")

  @staticmethod
  def new(mode, bound):
    return {
      "global": ClipMode.Global,
      "norm": ClipMode.Norm,
      None: ClipMode.Null
    }[mode](bound)


class LearningRate(object):
  class Cosine(object):
    def __init__(self, max, min, T_0, mul, lr_warmup_val, lr_warmup_steps):
      assert max is not None, "Need lr_max to use lr_cosine"
      assert min is not None, "Need lr_min to use lr_cosine"
      assert T_0 is not None, "Need lr_T_0 to use lr_cosine"
      assert mul is not None, "Need lr_T_mul to use lr_cosine"
      self.max = max
      self.min = min
      self.T_0 = T_0
      self.mul = mul
      self.lr_warmup_val = lr_warmup_val
      self.lr_warmup_steps = lr_warmup_steps
      self.last_reset = fw.Variable(0, dtype=fw.int64, name="last_reset")
      self.T_i = fw.Variable(self.T_0, dtype=fw.int64, name="T_i")

    def update(self, num_train_batches, train_step):
      assert num_train_batches is not None, "Need num_train_batches to use lr_cosine"

      curr_epoch = train_step // num_train_batches

      T_curr = curr_epoch - self.last_reset

      def _update():
        with fw.control_dependencies([
          self.last_reset.assign(curr_epoch, use_locking=True),
          self.T_i.assign(self.T_i * self.mul, use_locking=True)]):
          return self.min + 0.5 * (self.max - self.min) * (1.0 + fw.cos(fw.to_float(T_curr) / fw.to_float(self.T_i) * 3.1415926))

      def _no_update():
        return self.min + 0.5 * (self.max - self.min) * (1.0 + fw.cos(fw.to_float(T_curr) / fw.to_float(self.T_i) * 3.1415926))

      learning_rate = fw.cond(
        fw.greater_equal(T_curr, self.T_i),
        _update,
        _no_update)

      if self.lr_warmup_val is not None:
        return fw.cond(
          fw.less(train_step, self.lr_warmup_steps),
          lambda: self.lr_warmup_val,
          lambda: learning_rate)
      else:
        return learning_rate

  class Regular(object):
    def __init__(self, init, start, every, rate, dec_min, lr_warmup_val, lr_warmup_steps):
      fn = fw.exp_decay(init, every, rate, staircase=True)
      self.exp_decay = lambda train_step: fn(fw.maximum(train_step - start, 0))
      self.rate = rate
      self.dec_min = dec_min
      self.lr_warmup_val = lr_warmup_val
      self.lr_warmup_steps = lr_warmup_steps

    def update(self, _, train_step):
      learning_rate = self.exp_decay(train_step)
      if self.dec_min is not None:
        learning_rate = fw.maximum(learning_rate, self.dec_min)

      if self.lr_warmup_val is not None:
        return fw.cond(
          fw.less(train_step, self.lr_warmup_steps),
          lambda: self.lr_warmup_val,
          lambda: learning_rate)
      else:
        return learning_rate

  def __init__(self):
    raise AttributeError("Use factory method `new` instead.")

  @staticmethod
  def new(cosine, init, start, every, rate, dec_min, max, min, T_0, mul, lr_warmup_val=None, lr_warmup_steps=0):
    if True == cosine:
      return LearningRate.Cosine(max, min, T_0, mul, lr_warmup_val, lr_warmup_steps)
    elif False == cosine:
      return LearningRate.Regular(init, start, every, rate, dec_min, lr_warmup_val, lr_warmup_steps)
    else:
      raise KeyError("cosine must be True or False")

user_flags = []


def DEFINE_string(name, default_value, doc_string):
  flags.DEFINE_string(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def DEFINE_integer(name, default_value, doc_string):
  flags.DEFINE_integer(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def DEFINE_float(name, default_value, doc_string):
  flags.DEFINE_float(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def DEFINE_boolean(name, default_value, doc_string):
  flags.DEFINE_boolean(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def print_user_flags(line_limit=80):
  print("-" * 80)

  global user_flags
  FLAGS = flags.FLAGS

  for flag_name in sorted(user_flags):
    value = "{}".format(getattr(FLAGS, flag_name))
    log_string = flag_name
    log_string += "." * (line_limit - len(flag_name) - len(value))
    log_string += value
    print(log_string)


class Logger(object):
  def __init__(self, output_file):
    self.terminal = sys.stdout
    self.log = open(output_file, "a")

  def write(self, message):
    self.terminal.write(message)
    self.terminal.flush()
    self.log.write(message)
    self.log.flush()


def count_model_params(tf_variables):
  """
  Args:
    tf_variables: list of all model variables
  """

  num_vars = 0
  for var in tf_variables:
    num_vars += np.prod([dim for dim in var.get_shape()])
  return num_vars


class GradientCalculator(object):
  def __init__(self, l2_reg):

    def adjust(loss, tf_variables):
      l2_losses = []
      for var in tf_variables:
        l2_losses.append(fw.reduce_sum(var ** 2))
      return loss + l2_reg * fw.add_n(l2_losses)

    if l2_reg > 0:
      self.regularize = adjust
    else:
      self.regularize = lambda x, _: x # identity

  def __call__(self, loss, tf_variables, tape):
      return tape.gradient(self.regularize(loss, tf_variables), tf_variables) # Can only do this once


class TrainStep(object):
  def __init__(self, train_step, l2_reg, updater, clip_mode, num_train_batches, optim_algo, moving_average, get_grad_norms):
    self.train_step = train_step
    self.updater = updater
    self.clip_mode = clip_mode
    self.num_train_batches = num_train_batches
    self.grads = GradientCalculator(l2_reg)
    self.opt = optim_algo.get(self.learning_rate, moving_average)
    self.get_grad_norms = get_grad_norms

  def train_op(self, loss, vars, tape):
    self.train_step.assign_add(1)
    grads = self.grads(loss, vars, tape) # Can only do this once, so collect grad_norm(s) immediately
    grad_norm = fw.global_norm(grads)
    grad_norms = {}
    if self.get_grad_norms:
        for v, g in zip(vars, grads):
            if v is None or g is None:
                continue
            if isinstance(g, fw.IndexedSlices):
                grad_norms[v.name] = fw.sqrt(fw.reduce_sum(g.values ** 2))
            else:
                grad_norms[v.name] = fw.sqrt(fw.reduce_sum(g ** 2))
    return grad_norm, grad_norms, self.opt.apply_gradients(
      zip(self.clip_mode.clip(grads), vars),
      global_step=self.train_step)

  def learning_rate(self):
    return self.updater.update(self.num_train_batches, self.train_step)


def get_train_ops(
    train_step,
    updater,
    clip_mode=None,
    l2_reg=1e-4,
    num_train_batches=None,
    optim_algo=None,
    moving_average=None,
    get_grad_norms=False):
    """
    Args:
      clip_mode: "global", "norm", or None.
      moving_average: store the moving average of parameters
    """
    ts = TrainStep(train_step, l2_reg, updater, clip_mode, num_train_batches, optim_algo, moving_average, get_grad_norms)
    return ts.train_op, ts.learning_rate, ts.opt


class LayeredModel(object):
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
