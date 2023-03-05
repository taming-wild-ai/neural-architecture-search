import unittest
from unittest.mock import patch
from unittest import mock

import tensorflow as tf
import numpy as np

from src.cifar10.image_ops import drop_path, batch_norm, batch_norm_with_mask
from src.cifar10.child import DataFormat

class TestImageOps(unittest.TestCase):
    def test_drop_path(self):
        drop_path(tf.constant(np.ndarray((45000, 32, 32, 3)), dtype=tf.float32), keep_prob=0.9)

    @patch('src.cifar10.image_ops.fw.get_variable_scope')
    @patch('src.cifar10.image_ops.fw.control_dependencies')
    @patch('src.cifar10.image_ops.moving_averages', return_value="ama")
    @patch('src.cifar10.image_ops.fw.Constant', return_value="constant")
    @patch('src.cifar10.image_ops.fw.fused_batch_norm', return_value="fbn")
    @patch('src.cifar10.image_ops.fw.identity', return_value="identity")
    def test_batch_norm_nhwc_training(self, identity, fbn, constant, ama, cd, var_scope):
        var_scope().reuse = False
        mock_weights = mock.MagicMock()
        mock_weights.get = mock.MagicMock(return_value="get_variable")
        input_tensor = tf.constant(np.ndarray((45000, 32, 32, 3)))
        self.assertEqual("identity", batch_norm(input_tensor, True, DataFormat.new("NHWC"), mock_weights))
        mock_weights.get.assert_any_call("offset", [3], "constant", False)
        fbn.assert_called_with(input_tensor, "get_variable", "get_variable", epsilon=1e-5, data_format="NHWC", is_training=True)
        identity.assert_called_with("f")

    @patch('src.cifar10.image_ops.fw.get_variable_scope')
    @patch('src.cifar10.image_ops.fw.Constant', return_value="constant")
    @patch('src.cifar10.image_ops.fw.fused_batch_norm', return_value="fbn")
    @patch('src.cifar10.image_ops.fw.identity', return_value="identity")
    def test_batch_norm_nchw_not_training(self, identity, fbn, constant, var_scope):
        var_scope().reuse = False
        mock_weights = mock.MagicMock()
        mock_weights.get = mock.MagicMock(return_value="get_variable")
        input_tensor = tf.constant(np.ndarray((45000, 3, 32, 32)))
        self.assertEqual("f", batch_norm(input_tensor, False, DataFormat.new("NCHW"), mock_weights))
        mock_weights.get.assert_any_call("offset", [3], "constant", False)
        fbn.assert_called_with(input_tensor, "get_variable", "get_variable", mean="get_variable", variance="get_variable", epsilon=1e-5, data_format="NCHW", is_training=False)
        identity.assert_not_called()

    @patch('src.cifar10.image_ops.fw.control_dependencies')
    @patch('src.cifar10.image_ops.fw.identity', return_value="identity")
    @patch('src.cifar10.image_ops.fw.scatter_sub', return_value="scatter_sub")
    @patch('src.cifar10.image_ops.fw.fused_batch_norm', return_value=("fbn", 1.0, 1.0))
    @patch('src.cifar10.image_ops.fw.boolean_mask', return_value=1.0)
    @patch('src.cifar10.image_ops.fw.Constant', return_value="constant")
    @patch('src.cifar10.image_ops.fw.reshape')
    @patch('src.cifar10.image_ops.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.image_ops.fw.where', return_value="where")
    def test_batch_norm_with_mask_training(self, where, to_int32, reshape, constant, boolean_mask, fbn, scatter_sub, identity, cd):
        weights_mock = mock.MagicMock()
        self.assertEqual("identity", batch_norm_with_mask(None, True, None, 3, weights_mock))

    @patch('src.cifar10.image_ops.fw.get_variable_scope')
    @patch('src.cifar10.image_ops.fw.fused_batch_norm', return_value="fbn")
    @patch('src.cifar10.image_ops.fw.boolean_mask', return_value="boolean_mask")
    @patch('src.cifar10.image_ops.fw.Constant', return_value="constant")
    @patch('src.cifar10.image_ops.fw.reshape')
    @patch('src.cifar10.image_ops.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.image_ops.fw.where', return_value="where")
    def test_batch_norm_with_mask_not_training(self, where, to_int32, reshape, constant, boolean_mask, fbn, var_scope):
        var_scope().reuse = True
        weights_mock = mock.MagicMock()
        self.assertEqual('f', batch_norm_with_mask(None, False, None, 3, weights_mock))
        where.assert_called_with(None)
        to_int32.assert_called_with("where")
        reshape.assert_called_with("to_int32", [-1])
        weights_mock.get.assert_called_with("moving_variance", [3], "constant", True, trainable=False)
        weights_mock.get.assert_any_call("moving_mean", [3], "constant", True, trainable=False)
        weights_mock.get.assert_any_call("scale", [3], "constant", True)
        weights_mock.get.assert_any_call("offset", [3], "constant", True)
        boolean_mask.assert_called_with(weights_mock.get(), None)
        fbn.assert_called_with(None, "boolean_mask", "boolean_mask", mean="boolean_mask", variance="boolean_mask", epsilon=0.001, data_format="NHWC", is_training=False)

if "__main__" == __name__:
    unittest.main()