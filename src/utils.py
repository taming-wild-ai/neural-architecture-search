from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np

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

    def update(self, num_train_batches, train_step):
      assert num_train_batches is not None, ("Need num_train_batches to use"
                                           " lr_cosine")

      curr_epoch = train_step // num_train_batches

      last_reset = fw.Variable(0, dtype=fw.int64, name="last_reset")
      T_i = fw.Variable(self.T_0, dtype=fw.int64, name="T_i")
      T_curr = curr_epoch - last_reset

      def _update():
        with fw.control_dependencies([
          last_reset.assign(curr_epoch, use_locking=True),
          T_i.assign(T_i * self.mul, use_locking=True)]):
          return self.min + 0.5 * (self.max - self.min) * (1.0 + fw.cos(fw.to_float(T_curr) / fw.to_float(T_i) * 3.1415926))

      def _no_update():
        return self.min + 0.5 * (self.max - self.min) * (1.0 + fw.cos(fw.to_float(T_curr) / fw.to_float(T_i) * 3.1415926))


      learning_rate = fw.cond(
          fw.greater_equal(T_curr, T_i), _update, _no_update)

      if self.lr_warmup_val is not None:
        return fw.cond(
          fw.less(train_step, self.lr_warmup_steps),
          lambda: self.lr_warmup_val,
          lambda: learning_rate)
      else:
        return learning_rate

  class Regular(object):
    def __init__(self, init, start, every, rate, dec_min, lr_warmup_val, lr_warmup_steps):
      self.init = init
      self.start = start
      self.every = every
      self.rate = rate
      self.dec_min = dec_min
      self.lr_warmup_val = lr_warmup_val
      self.lr_warmup_steps = lr_warmup_steps

    def update(self, _, train_step):
      learning_rate = fw.exp_decay(
        self.init, fw.maximum(train_step - self.start, 0), self.every,
        self.rate, staircase=True)
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
  fw.DEFINE_string(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def DEFINE_integer(name, default_value, doc_string):
  fw.DEFINE_integer(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def DEFINE_float(name, default_value, doc_string):
  fw.DEFINE_float(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def DEFINE_boolean(name, default_value, doc_string):
  fw.DEFINE_boolean(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def print_user_flags(line_limit=80):
  print("-" * 80)

  global user_flags
  FLAGS = fw.FLAGS

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


class L2Reg(object):
  def __init__(self, l2_reg):

    def adjust(loss, tf_variables):
      l2_losses = []
      for var in tf_variables:
        l2_losses.append(fw.reduce_sum(var ** 2))
      return loss + l2_reg * fw.add_n(l2_losses)

    if l2_reg > 0:
      self.layer = adjust
    else:
      self.layer = lambda x, _: x # identity

  def __call__(self, loss, tf_variables):
    return self.layer(loss, tf_variables)


def get_train_ops(
    loss,
    tf_variables,
    train_step,
    updater,
    clip_mode=None,
    l2_reg=1e-4,
    lr_warmup_val=None,
    lr_warmup_steps=100,
    num_train_batches=None,
    optim_algo=None,
    moving_average=None,
    get_grad_norms=False):
  """
  Args:
    clip_mode: "global", "norm", or None.
    moving_average: store the moving average of parameters
  """
  loss_adjuster = L2Reg(l2_reg)

  grads = fw.gradients(loss_adjuster(loss, tf_variables), tf_variables)
  grad_norm = fw.global_norm(grads)

  learning_rate = updater.update(num_train_batches, train_step)
  opt = optim_algo.get(learning_rate, moving_average)
  train_op = opt.apply_gradients(
    zip(clip_mode.clip(grads), tf_variables),
    global_step=train_step)

  if get_grad_norms:
    grad_norms = {}
    for v, g in zip(tf_variables, grads):
      if v is None or g is None:
        continue
      if isinstance(g, fw.IndexedSlices):
        grad_norms[v.name] = fw.sqrt(fw.reduce_sum(g.values ** 2))
      else:
        grad_norms[v.name] = fw.sqrt(fw.reduce_sum(g ** 2))
    return train_op, learning_rate, grad_norm, opt, grad_norms
  else:
    return train_op, learning_rate, grad_norm, opt


class LayeredModel(object):
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
