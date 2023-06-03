import unittest
import unittest.mock as mock
from unittest.mock import patch

from src.cifar10.micro_controller import MicroController

import tensorflow as tf
from absl import flags
flags.FLAGS(['test'])

class TestMicroController(unittest.TestCase):
    @patch('src.cifar10.micro_controller.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.micro_controller.fw.reshape', return_value="reshape")
    @patch('src.cifar10.micro_controller.fw.matmul', return_value="matmul")
    @patch('src.cifar10.micro_controller.fw.TensorArray')
    @patch('src.cifar10.micro_controller.stack_lstm', return_value=([1], [2]))
    @patch('src.cifar10.micro_controller.fw.constant_initializer')
    @patch('src.cifar10.micro_controller.fw.Variable', return_value="gv")
    @patch('src.cifar10.micro_controller.fw.random_uniform_initializer')
    @patch('src.cifar10.micro_controller.fw.zeros', return_value="zeros")
    @patch('src.cifar10.micro_controller.print')
    def test_constructor(self, print, zeros, rui, gv, const, stack_lstm, tensor_array, matmul, reshape, reduce_sum):
        rui.__call__ = mock.MagicMock(return_value='rui')
        mock_tensor_array = mock.MagicMock()
        mock_tensor_array.write = mock.MagicMock(return_value=mock_tensor_array)
        mock_tensor_array.stack = mock.MagicMock(return_value="stack")
        tensor_array.return_value = mock_tensor_array
        with patch('src.cifar10.micro_controller.fw.while_loop', return_value=mock_tensor_array) as while_loop:
            with tf.Graph().as_default():
                mc = MicroController(temperature=1.0, tanh_constant=1.0, op_tanh_reduce=1.0)
                logits1, c, h = mc.sample_logit1(None, None)
                logits2, _c, _h = mc.sample_logit2(c, h)
                mc.generate_sample_arc()
                mc.sample_entropy(logits1, logits2)
                mc.sample_log_prob(logits1, logits2)
                print.assert_any_call('-' * 80)
                print.assert_any_call("Building ConvController")
                zeros.assert_any_call([1, 32], tf.float32)
                zeros.assert_called_with([1, 32], tf.float32)
                rui().assert_called_with([32, 1])
                gv.assert_called_with(rui()(), 'v', import_scope='controller/attention/', trainable=True)
                stack_lstm.assert_called_with('gv', [1], [2], ['gv', 'gv'])
                mock_tensor_array.write.assert_called_with(1, 'matmul')
                matmul.assert_called_with(2, 'gv')
                while_loop.assert_called()
                reshape.assert_any_call(tensor_array().__getitem__().stack(), [-1])
                reshape.assert_called_with(tensor_array().__getitem__().stack(), [-1])
                reduce_sum.assert_called_with(tensor_array().__getitem__())

    @patch('src.cifar10.micro_controller.fw.control_dependencies')
    @patch('src.cifar10.micro_controller.fw.to_float', return_value=1.0)
    @patch('src.cifar10.micro_controller.fw.reduce_sum', return_value=1.0)
    @patch('src.cifar10.micro_controller.fw.reshape', return_value="reshape")
    @patch('src.cifar10.micro_controller.fw.matmul', return_value="matmul")
    @patch('src.cifar10.micro_controller.fw.TensorArray')
    @patch('src.cifar10.micro_controller.stack_lstm', return_value=([1], [2]))
    @patch('src.cifar10.micro_controller.fw.constant_initializer')
    @patch('src.cifar10.micro_controller.fw.Variable')
    @patch('src.cifar10.micro_controller.fw.random_uniform_initializer')
    @patch('src.cifar10.micro_controller.get_train_ops')
    @patch('src.cifar10.micro_controller.fw.zeros', return_value="zeros")
    @patch('src.cifar10.micro_controller.print')
    def test_build_trainer(self, print, zeros, get_train_ops, rui, variable, const, stack_lstm, tensor_array, matmul, reshape, reduce_sum, to_float, cd):
        train_op = mock.MagicMock(name='train_op', return_value='train_op')
        grad_norm = mock.MagicMock(name='grad_norm', return_value='grad_norm')
        get_train_ops.return_value = (train_op, 2, grad_norm, 4)
        variable().assign_sub = mock.MagicMock(return_value='assign_sub')
        rui.__call__ = mock.MagicMock(return_value="rui")
        mock_tensor_array = mock.MagicMock()
        mock_tensor_array.write = mock.MagicMock(return_value=mock_tensor_array)
        mock_tensor_array.stack = mock.MagicMock(return_value="stack")
        tensor_array.return_value = mock_tensor_array
        with patch('src.cifar10.micro_controller.fw.while_loop', return_value=mock_tensor_array) as while_loop:
            with tf.Graph().as_default() as graph:
                variable()._as_graph_element().graph = graph
                mc = MicroController(temperature=1.0, tanh_constant=1.0, op_tanh_reduce=1.0, entropy_weight=1.0)
                logits1, c, h = mc.sample_logit1(None, None)
                logits2, _c, _h = mc.sample_logit2(c, h)
                mc.generate_sample_arc()
                mc.sample_entropy(logits1, logits2)
                mc.sample_log_prob(logits1, logits2)
                mock_child = mock.MagicMock(name='mock_child')
                shuffle = mock.MagicMock(return_value=('x_valid_shuffle', 'y_valid_shuffle'))
                vrl = mock.MagicMock(return_value='vrl')
                self.assertEqual((train_op, 2, grad_norm, 4), mc.build_trainer(mock_child, vrl))
                train_op(mc.loss, [])
                grad_norm(mc.loss, [])
                mc.loss('logits', 'y_valid_shuffle')
                zeros.assert_called_with([1, 32], tf.float32)
                variable().assign_sub.assert_called_with(variable().__sub__().__rmul__())
                get_train_ops.assert_called_with(
                    variable(),
                    mc.learning_rate,
                    clip_mode=mc.clip_mode,
                    optim_algo=mc.optim_algo)
                train_op.assert_called_with(mc.loss, [])
                grad_norm.assert_called_with(mc.loss, [])
                to_float.assert_called_with(mock_child.batch_size)
                cd.assert_called_with(['assign_sub'])
                while_loop.assert_called()

if "__main__" == __name__:
    unittest.main()