from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np

import src.framework as fw

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


class TextColors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'


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


def get_train_ops(
    loss,
    tf_variables,
    train_step,
    clip_mode=None,
    grad_bound=None,
    l2_reg=1e-4,
    lr_warmup_val=None,
    lr_warmup_steps=100,
    lr_init=0.1,
    lr_dec_start=0,
    lr_dec_every=10000,
    lr_dec_rate=0.1,
    lr_dec_min=None,
    lr_cosine=False,
    lr_max=None,
    lr_min=None,
    lr_T_0=None,
    lr_T_mul=None,
    num_train_batches=None,
    optim_algo=None,
    sync_replicas=False,
    num_aggregate=None,
    num_replicas=None,
    get_grad_norms=False,
    moving_average=None):
  """
  Args:
    clip_mode: "global", "norm", or None.
    moving_average: store the moving average of parameters
  """

  if l2_reg > 0:
    l2_losses = []
    for var in tf_variables:
      l2_losses.append(fw.reduce_sum(var ** 2))
    l2_loss = fw.add_n(l2_losses)
    loss += l2_reg * l2_loss

  grads = fw.gradients(loss, tf_variables)
  grad_norm = fw.global_norm(grads)

  grad_norms = {}
  for v, g in zip(tf_variables, grads):
    if v is None or g is None:
      continue
    if isinstance(g, fw.IndexedSlices):
      grad_norms[v.name] = fw.sqrt(fw.reduce_sum(g.values ** 2))
    else:
      grad_norms[v.name] = fw.sqrt(fw.reduce_sum(g ** 2))

  if clip_mode is not None:
    assert grad_bound is not None, "Need grad_bound to clip gradients."
    if clip_mode == "global":
      grads, _ = fw.clip_by_global_norm(grads, grad_bound)
    elif clip_mode == "norm":
      clipped = []
      for g in grads:
        if isinstance(g, fw.IndexedSlices):
          c_g = fw.clip_by_norm(g.values, grad_bound)
          c_g = fw.IndexedSlices(g.indices, c_g)
        else:
          c_g = fw.clip_by_norm(g, grad_bound)
        clipped.append(g)
      grads = clipped
    else:
      raise NotImplementedError("Unknown clip_mode {}".format(clip_mode))

  if lr_cosine:
    assert lr_max is not None, "Need lr_max to use lr_cosine"
    assert lr_min is not None, "Need lr_min to use lr_cosine"
    assert lr_T_0 is not None, "Need lr_T_0 to use lr_cosine"
    assert lr_T_mul is not None, "Need lr_T_mul to use lr_cosine"
    assert num_train_batches is not None, ("Need num_train_batches to use"
                                           " lr_cosine")

    curr_epoch = train_step // num_train_batches

    last_reset = fw.Variable(0, dtype=fw.int64, name="last_reset")
    T_i = fw.Variable(lr_T_0, dtype=fw.int64, name="T_i")
    T_curr = curr_epoch - last_reset

    def _update():
      update_last_reset = fw.assign(last_reset, curr_epoch)
      update_T_i = fw.assign(T_i, T_i * lr_T_mul)
      with fw.control_dependencies([update_last_reset, update_T_i]):
        rate = fw.to_float(T_curr) / fw.to_float(T_i) * 3.1415926
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + fw.cos(rate))
      return lr

    def _no_update():
      rate = fw.to_float(T_curr) / fw.to_float(T_i) * 3.1415926
      lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + fw.cos(rate))
      return lr

    learning_rate = fw.cond(
      fw.greater_equal(T_curr, T_i), _update, _no_update)
  else:
    learning_rate = fw.exp_decay(
      lr_init, fw.maximum(train_step - lr_dec_start, 0), lr_dec_every,
      lr_dec_rate, staircase=True)
    if lr_dec_min is not None:
      learning_rate = fw.maximum(learning_rate, lr_dec_min)

  if lr_warmup_val is not None:
    learning_rate = fw.cond(fw.less(train_step, lr_warmup_steps),
                            lambda: lr_warmup_val, lambda: learning_rate)

  # if get_grad_norms:
  #   g_1, g_2 = 0.0001, 0.0001
  #   for v, g in zip(tf_variables, grads):
  #     if g is not None:
  #       if isinstance(g, fw.IndexedSlices):
  #         g_n = fw.reduce_sum(g.values ** 2)
  #       else:
  #         g_n = fw.reduce_sum(g ** 2)
  #       if "enas_cell" in v.name:
  #         print("g_1: {}".format(v.name))
  #         g_1 += g_n
  #       else:
  #         print("g_2: {}".format(v.name))
  #         g_2 += g_n
  #   learning_rate = fw.Print(learning_rate, [g_1, g_2, fw.sqrt(g_1 / g_2)],
  #                            message="g_1, g_2, g_1/g_2: ", summarize=5)

  if optim_algo == "momentum":
    opt = fw.Optimizer.Momentum(learning_rate)
  elif optim_algo == "sgd":
    opt = fw.Optimizer.SGD(learning_rate)
  elif optim_algo == "adam":
    opt = fw.Optimizer.Adam(learning_rate)
  else:
    raise ValueError("Unknown optim_algo {}".format(optim_algo))

  if sync_replicas:
    assert num_aggregate is not None, "Need num_aggregate to sync."
    assert num_replicas is not None, "Need num_replicas to sync."

    opt = fw.Optimizer.SyncReplicas(opt, num_aggregate, num_replicas)

  if moving_average is not None:
    opt = fw.Optimizer.MovingAverage(opt, moving_average)

  train_op = opt.apply_gradients(
    zip(grads, tf_variables), global_step=train_step)

  if get_grad_norms:
    return train_op, learning_rate, grad_norm, opt, grad_norms
  else:
    return train_op, learning_rate, grad_norm, opt

