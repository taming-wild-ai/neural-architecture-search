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

  if l2_reg > 0:
    l2_losses = []
    for var in tf_variables:
      l2_losses.append(fw.reduce_sum(var ** 2))
    loss += l2_reg * fw.add_n(l2_losses)

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

  grads = clip_mode.clip(grads)

  learning_rate = updater.update(num_train_batches, train_step)

  if lr_warmup_val is not None:
    learning_rate = fw.cond(fw.less(train_step, lr_warmup_steps),
                            lambda: lr_warmup_val, lambda: learning_rate)
  opt = optim_algo.get(learning_rate, moving_average)
  train_op = opt.apply_gradients(
    zip(grads, tf_variables), global_step=train_step)

  if get_grad_norms:
    return train_op, learning_rate, grad_norm, opt, grad_norms
  else:
    return train_op, learning_rate, grad_norm, opt

