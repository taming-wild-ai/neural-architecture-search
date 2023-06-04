import unittest
from unittest.mock import patch
from unittest import mock

import tensorflow as tf
import numpy as np

from src.cifar10.image_ops import drop_path, BatchNorm, BatchNormWithMask
from src.cifar10.child import DataFormat

class TestImageOps(unittest.TestCase):
    def test_drop_path(self):
        drop_path(tf.constant(np.ndarray((45000, 32, 32, 3)), dtype=tf.float32), keep_prob=0.9)

    @patch('src.cifar10.image_ops.fw.reshape')
    @patch('src.cifar10.image_ops.fw.control_dependencies')
    @patch('src.cifar10.image_ops.moving_averages', return_value="ama")
    @patch('src.cifar10.image_ops.fw.constant_initializer', return_value="constant")
    @patch('src.cifar10.image_ops.fw.fused_batch_norm', return_value=("fbn", 1.0, 1.0, 1.0, 1.0))
    @patch('src.cifar10.image_ops.fw.identity', return_value="identity")
    def test_batch_norm_nhwc_training(self, identity, fbn, constant, ama, cd, reshape):
        mock_weights = mock.MagicMock()
        mock_weights.get = mock.MagicMock(return_value="get_variable")
        input_tensor = tf.constant(np.ndarray((45000, 32, 32, 3)))
        with tf.Graph().as_default():
            bn = BatchNorm(True, DataFormat.new('NHWC'), mock_weights, 3, True)
            self.assertEqual("identity", bn(input_tensor))
            mock_weights.get.assert_any_call(True, 'bn/', "offset", [3], "constant")
            fbn.assert_called_with(x=input_tensor, scale="get_variable", offset="get_variable", mean=reshape(), variance=reshape(), epsilon=1e-5, data_format="NHWC", is_training=True)
            identity.assert_called_with("fbn")

    @patch('src.cifar10.image_ops.fw.constant_initializer', return_value="constant")
    @patch('src.cifar10.image_ops.fw.fused_batch_norm', return_value="fbn")
    @patch('src.cifar10.image_ops.fw.identity', return_value="identity")
    def test_batch_norm_nchw_not_training(self, identity, fbn, constant):
        mock_weights = mock.MagicMock()
        mock_weights.get = mock.MagicMock(return_value="get_variable")
        input_tensor = tf.constant(np.ndarray((45000, 3, 32, 32)))
        bn = BatchNorm(False, DataFormat.new('NCHW'), mock_weights, 3, True)
        self.assertEqual("f", bn(input_tensor))
        mock_weights.get.assert_any_call(True, 'bn/', "offset", [3], "constant")
        mock_weights.get.assert_called_with(True, 'bn/', 'moving_variance', [3], 'constant', trainable=False)
        fbn.assert_called_with(input_tensor, "get_variable", "get_variable", mean="get_variable", variance="get_variable", epsilon=1e-5, data_format="NCHW", is_training=False)
        identity.assert_not_called()

    @patch('src.cifar10.image_ops.fw.control_dependencies')
    @patch('src.cifar10.image_ops.fw.identity', return_value="identity")
    @patch('src.cifar10.image_ops.fw.scatter_sub', return_value="scatter_sub")
    @patch('src.cifar10.image_ops.fw.fused_batch_norm', return_value=("fbn", 1.0, 1.0, 1.0, 1.0))
    @patch('src.cifar10.image_ops.fw.boolean_mask', return_value=1.0)
    @patch('src.cifar10.image_ops.fw.constant_initializer', return_value="constant")
    @patch('src.cifar10.image_ops.fw.reshape')
    @patch('src.cifar10.image_ops.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.image_ops.fw.where', return_value="where")
    def test_batch_norm_with_mask_training(self, where, to_int32, reshape, constant, boolean_mask, fbn, scatter_sub, identity, cd):
        weights_mock = mock.MagicMock()
        bn = BatchNormWithMask(True, None, 3, weights_mock, True)
        self.assertEqual("identity", bn(None))

    @patch('src.cifar10.image_ops.fw.fused_batch_norm', return_value=("fbn", 1.0, 1.0, 1.0, 1.0))
    @patch('src.cifar10.image_ops.fw.boolean_mask', return_value="boolean_mask")
    @patch('src.cifar10.image_ops.fw.constant_initializer', return_value="constant")
    @patch('src.cifar10.image_ops.fw.reshape')
    @patch('src.cifar10.image_ops.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.image_ops.fw.where', return_value="where")
    def test_batch_norm_with_mask_not_training(self, where, to_int32, reshape, constant, boolean_mask, fbn):
        weights_mock = mock.MagicMock()
        with tf.Graph().as_default():
            bn = BatchNormWithMask(False, None, 3, weights_mock, True)
            self.assertEqual('fbn', bn(None))
            where.assert_called_with(None)
            to_int32.assert_called_with("where")
            reshape.assert_called_with("to_int32", [-1])
            weights_mock.get.assert_called_with(True, 'bn/', "moving_variance", [3], "constant", trainable=False)
            weights_mock.get.assert_any_call(True, 'bn/', "moving_mean", [3], "constant", trainable=False)
            weights_mock.get.assert_any_call(True, 'bn/', "scale", [3], "constant")
            weights_mock.get.assert_any_call(True, 'bn/', "offset", [3], "constant")
            boolean_mask.assert_called_with(weights_mock.get(), None)
            fbn.assert_called_with(x=None, scale='boolean_mask', offset='boolean_mask', mean='boolean_mask', variance='boolean_mask', epsilon=0.001, data_format='NHWC', is_training=False)

if "__main__" == __name__:
    unittest.main()