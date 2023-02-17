import numpy as np
import src.framework as fw

def lstm(x, prev_c, prev_h, w):
  ifog = fw.matmul(fw.concat([x, prev_h], axis=1), w)
  i, f, o, g = fw.split(ifog, 4, axis=1)
  i = fw.sigmoid(i)
  f = fw.sigmoid(f)
  o = fw.sigmoid(o)
  g = fw.tanh(g)
  next_c = i * g + f * prev_c
  next_h = o * fw.tanh(next_c)
  return next_c, next_h


def stack_lstm(x, prev_c, prev_h, w):
  next_c, next_h = [], []
  for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
    inputs = x if layer_id == 0 else next_h[-1]
    curr_c, curr_h = lstm(inputs, _c, _h, _w)
    next_c.append(curr_c)
    next_h.append(curr_h)
  return next_c, next_h
