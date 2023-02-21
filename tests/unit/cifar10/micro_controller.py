import unittest
import unittest.mock as mock
from unittest.mock import patch

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

from src.cifar10.micro_controller import MicroController

class TestMicroController(unittest.TestCase):
    @patch('src.cifar10.micro_controller.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.micro_controller.fw.reshape', return_value="reshape")
    @patch('src.cifar10.micro_controller.fw.matmul', return_value="matmul")
    @patch('src.cifar10.micro_controller.fw.TensorArray')
    @patch('src.cifar10.micro_controller.stack_lstm', return_value=([1], [2]))
    @patch('src.cifar10.micro_controller.fw.Constant')
    @patch('src.cifar10.micro_controller.fw.get_variable', return_value="gv")
    @patch('src.cifar10.micro_controller.fw.random_uniform_initializer', return_value="rui")
    @patch('src.cifar10.micro_controller.fw.zeros', return_value="zeros")
    @patch('src.cifar10.micro_controller.print')
    def test_constructor(self, print, zeros, rui, gv, const, stack_lstm, tensor_array, matmul, reshape, reduce_sum):
        mock_tensor_array = mock.MagicMock()
        mock_tensor_array.write = mock.MagicMock(return_value=mock_tensor_array)
        mock_tensor_array.stack = mock.MagicMock(return_value="stack")
        tensor_array.return_value = mock_tensor_array
        with patch('src.cifar10.micro_controller.fw.while_loop', return_value=mock_tensor_array) as while_loop:
            mc = MicroController(temperature=1.0, tanh_constant=1.0, op_tanh_reduce=1.0)
            print.assert_any_call('-' * 80)
            print.assert_any_call("Building ConvController")
            zeros.assert_called_with([1, 32], tf.float32)
            rui.assert_called_with(minval=-0.1, maxval=0.1)
            gv.assert_called_with('v', [32, 1])
            stack_lstm.assert_called_with('gv', [1], [2], ['gv', 'gv'])
            mock_tensor_array.write.assert_called_with(1, 'matmul')
            matmul.assert_called_with(2, 'gv')
            while_loop.assert_called()
            reshape.assert_called()
            reduce_sum.assert_called()

    @patch('src.cifar10.micro_controller.fw.control_dependencies')
    @patch('src.cifar10.micro_controller.fw.to_float', return_value=1.0)
    @patch('src.cifar10.micro_controller.fw.reduce_sum', return_value=1.0)
    @patch('src.cifar10.micro_controller.fw.reshape', return_value="reshape")
    @patch('src.cifar10.micro_controller.fw.matmul', return_value="matmul")
    @patch('src.cifar10.micro_controller.fw.TensorArray')
    @patch('src.cifar10.micro_controller.stack_lstm', return_value=([1], [2]))
    @patch('src.cifar10.micro_controller.fw.Constant')
    @patch('src.cifar10.micro_controller.fw.get_variable', return_value="gv")
    @patch('src.cifar10.micro_controller.fw.random_uniform_initializer', return_value="rui")
    @patch('src.cifar10.micro_controller.get_train_ops', return_value=(1, 2, 3, 4))
    @patch('src.cifar10.micro_controller.fw.assign_sub', return_value="assign_sub")
    @patch('src.cifar10.micro_controller.fw.zeros', return_value="zeros")
    @patch('src.cifar10.micro_controller.print')
    def test_build_trainer(self, print, zeros, assign_sub, get_train_ops, rui, get_variable, const, stack_lstm, tensor_array, matmul, reshape, reduce_sum, to_float, cd):
        mock_tensor_array = mock.MagicMock()
        mock_tensor_array.write = mock.MagicMock(return_value=mock_tensor_array)
        mock_tensor_array.stack = mock.MagicMock(return_value="stack")
        tensor_array.return_value = mock_tensor_array
        with patch('src.cifar10.micro_controller.fw.while_loop', return_value=mock_tensor_array) as while_loop:
            mc = MicroController(temperature=1.0, tanh_constant=1.0, op_tanh_reduce=1.0, entropy_weight=1.0)
            mock_child = mock.MagicMock()
            mc.build_trainer(mock_child)
            mock_child.build_valid_rl.assert_called_with()
            zeros.assert_called_with([1, 32], tf.float32)
            assign_sub.assert_called()
            get_train_ops.assert_called()
            to_float.assert_called()
            cd.assert_called_with(['assign_sub'])

if "__main__" == __name__:
    unittest.main()