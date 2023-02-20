import numpy as np
import src.framework as fw

def lstm(x, prev_c, prev_h, w):
  i, f, o, g = fw.split(
    fw.matmul(
      fw.concat([x, prev_h], axis=1),
      w),
    4,
    axis=1)
  next_c = fw.sigmoid(i) * fw.tanh(g) + fw.sigmoid(f) * prev_c
  return next_c, fw.sigmoid(o) * fw.tanh(next_c)


def stack_lstm(x, prev_c, prev_h, w):
  next_c, next_h = [], []
  for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
    curr_c, curr_h = lstm(x if layer_id == 0 else next_h[-1], _c, _h, _w)
    next_c.append(curr_c)
    next_h.append(curr_h)
  return next_c, next_h
