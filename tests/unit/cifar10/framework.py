import sys
import unittest

import tensorflow as tf


class TestFrameworkWrapper(unittest.TestCase):
    def test_scatter_sub(self):
        SIZE = 36
        shape = [SIZE]
        moving_mean1 = tf.Variable([0.0] * SIZE, shape=shape)
        moving_mean2 = tf.Variable([0.0] * SIZE, shape=shape)
        mask = [True] * SIZE
        indices1 = tf.reshape(tf.cast(tf.where(mask), tf.int32), [-1])
        res1 = tf.compat.v1.scatter_sub(moving_mean1, indices1, [0.5] * SIZE).value()
        indices2 = tf.cast(tf.where(mask), tf.int32) # NOTE Required difference from indices1
        res2 = tf.tensor_scatter_nd_sub(moving_mean2, indices2, [0.5] * SIZE)
        result = True
        for el in (res1 == res2):
            if not el:
                result = False
                break
        self.assertTrue(result)
