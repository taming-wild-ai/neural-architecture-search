import unittest
import unittest.mock as mock
from unittest.mock import patch

import numpy as np
import src.framework as fw

from src.cifar10.macro_controller import MacroController
from src.cifar10.macro_child import DEFINE_integer # for child_num_layers, child_num_branches, child_out_filters

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class TestMacroController(unittest.TestCase):
    @patch('src.cifar10.macro_controller.fw.random_uniform_initializer')
    @patch('src.cifar10.macro_controller.fw.to_float', return_value=tf.constant(np.ones((1, 2))))
    @patch('src.cifar10.macro_controller.fw.reduce_sum', return_value=4.0)
    @patch('src.cifar10.macro_controller.fw.reduce_mean', return_value='reduce_mean')
    @patch('src.cifar10.macro_controller.fw.stack', return_value='stack')
    @patch('src.cifar10.macro_controller.fw.tanh', return_value=1.0)
    @patch('src.cifar10.macro_controller.fw.concat', return_value=2.0)
    @patch('src.cifar10.macro_controller.fw.embedding_lookup', return_value="embedding_lookup")
    @patch('src.cifar10.macro_controller.fw.stop_gradient', return_value="stop_gradient")
    @patch('src.cifar10.macro_controller.fw.sparse_softmax_cross_entropy_with_logits', return_value=0.5)
    @patch('src.cifar10.macro_controller.fw.reshape', return_value=3.0)
    @patch('src.cifar10.macro_controller.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.macro_controller.fw.multinomial', return_value="multinomial")
    @patch('src.cifar10.macro_controller.stack_lstm', return_value=([1.0], [2.0]))
    @patch('src.cifar10.macro_controller.fw.matmul', return_value=np.ones((5)))
    @patch('src.cifar10.macro_controller.fw.Variable', return_value="get_variable")
    @patch('src.cifar10.macro_controller.fw.zeros', return_value="zeros")
    @patch('src.cifar10.macro_controller.print')
    def test_constructor_not_whole_channels(self, print, zeros, get_variable, matmul, stack_lstm, multinomial, to_int32, reshape, sscewl, stop_gradient, embedding_lookup, concat, tanh, stack, reduce_mean, reduce_sum, to_float, rui):
        rui.__call__ = mock.MagicMock(return_value='rui')
        with tf.Graph().as_default():
            mc = MacroController(temperature=0.9)
            self.assertEqual(MacroController, type(mc))
            print.assert_any_call('-' * 80)
            zeros.assert_called_with([1, 32], tf.float32)
            rui().assert_called_with([32, 1])
            get_variable.assert_called_with(rui()(), name='v', trainable=True)
            matmul.assert_called_with(2.0, 'get_variable')
            stack_lstm.assert_called_with('embedding_lookup', [1.0], [2.0], ['get_variable', 'get_variable'])
            multinomial.assert_called_with(0.5, 1)
            to_int32.assert_called_with('multinomial')
            reshape.assert_called_with(2.0, [-1])
            sscewl.assert_called_with(logits=0.5, labels=3.0)
            stop_gradient.assert_called_with(4.0)
            embedding_lookup.assert_called_with('get_variable', 3.0)
            concat.assert_called_with([3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0], axis=0)
            tanh.assert_called_with(2.0)
            reduce_sum.assert_called_with('stack')
            to_float.assert_called_with(3.0)

    @patch('src.cifar10.macro_controller.fw.random_uniform_initializer')
    @patch('src.cifar10.macro_controller.fw.to_float', return_value=tf.constant(np.ones((1, 2))))
    @patch('src.cifar10.macro_controller.fw.reduce_mean', return_value='reduce_mean')
    @patch('src.cifar10.macro_controller.fw.reduce_sum', return_value=4.0)
    @patch('src.cifar10.macro_controller.fw.stack', return_value='stack')
    @patch('src.cifar10.macro_controller.fw.tanh', return_value=1.0)
    @patch('src.cifar10.macro_controller.fw.concat', return_value=2.0)
    @patch('src.cifar10.macro_controller.fw.embedding_lookup', return_value="embedding_lookup")
    @patch('src.cifar10.macro_controller.fw.stop_gradient', return_value="stop_gradient")
    @patch('src.cifar10.macro_controller.fw.sparse_softmax_cross_entropy_with_logits', return_value=0.5)
    @patch('src.cifar10.macro_controller.fw.reshape', return_value=3.0)
    @patch('src.cifar10.macro_controller.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.macro_controller.fw.multinomial', return_value="multinomial")
    @patch('src.cifar10.macro_controller.stack_lstm', return_value=([1.0], [2.0]))
    @patch('src.cifar10.macro_controller.fw.matmul', return_value=np.ones((5)))
    @patch('src.cifar10.macro_controller.fw.Variable', return_value="get_variable")
    @patch('src.cifar10.macro_controller.fw.zeros', return_value="zeros")
    @patch('src.cifar10.macro_controller.print')
    def test_constructor_whole_channels(self, print, zeros, get_variable, matmul, stack_lstm, multinomial, to_int32, reshape, sscewl, stop_gradient, embedding_lookup, concat, tanh, stack, reduce_sum, reduce_mean, to_float, rui):
        rui.__call__ = mock.MagicMock(return_value='rui')
        fw.FLAGS.controller_search_whole_channels = True
        fw.FLAGS.child_num_layers = 4
        fw.FLAGS.child_num_branches = 6
        fw.FLAGS.child_out_filters = 24
        fw.FLAGS.controller_tanh_constant = 0.5
        with tf.Graph().as_default():
            self.assertEqual(MacroController, type(MacroController(temperature=0.9)))
        print.assert_any_call('-' * 80)
        zeros.assert_called_with([1, 32], tf.float32)
        rui().assert_called_with([32, 1])
        get_variable.assert_called_with(rui()(), name='v', trainable=True)
        matmul.assert_called_with(2.0, 'get_variable')
        stack_lstm.assert_called_with('embedding_lookup', [1.0], [2.0], ['get_variable', 'get_variable'])
        multinomial.assert_called_with(0.5, 1)
        to_int32.assert_called_with('multinomial')
        reshape.assert_called_with(2.0, [-1])
        sscewl.assert_called_with(logits=0.5, labels=3.0)
        stop_gradient.assert_called_with(4.0)
        embedding_lookup.assert_called_with('get_variable', 3.0)
        concat.assert_called_with([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], axis=0)
        tanh.assert_called_with(2.0)
        reduce_sum.assert_called_with('stack')
        to_float.assert_called_with(3.0)

    @patch('src.cifar10.macro_controller.fw.control_dependencies')
    @patch('src.cifar10.macro_controller.fw.reduce_sum', return_value=4.0)
    @patch('src.cifar10.macro_controller.fw.concat', return_value=2.0)
    @patch('src.cifar10.macro_controller.fw.fill', return_value="fill")
    @patch('src.cifar10.macro_controller.fw.where', return_value="where")
    @patch('src.cifar10.macro_controller.fw.less_equal', return_value="less_equal")
    @patch('src.cifar10.macro_controller.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.macro_controller.fw.multinomial', return_value="multinomial")
    @patch('src.cifar10.macro_controller.fw.tanh', return_value=0.0)
    @patch('src.cifar10.macro_controller.fw.matmul')
    @patch('src.cifar10.macro_controller.stack_lstm', return_value=([1.0], [2.0]))
    @patch('src.cifar10.macro_controller.fw.embedding_lookup', return_value="embedding_lookup")
    @patch('src.cifar10.macro_controller.fw.sparse_softmax_cross_entropy_with_logits', return_value=0.5)
    @patch('src.cifar10.macro_controller.fw.reshape', return_value=3.0)
    @patch('src.cifar10.macro_controller.fw.to_float', return_value=tf.constant(np.ones((1, 2))))
    @patch('src.cifar10.macro_controller.get_train_ops')
    @patch('src.cifar10.macro_controller.fw.Variable')
    @patch('src.cifar10.macro_controller.fw.zeros', return_value="zeros")
    @patch('src.cifar10.macro_controller.print')
    def test_build_trainer(self, print, zeros, variable, get_train_ops, to_float, reshape, sscewl, embedding_lookup, stack_lstm, matmul, tanh, multinomial, to_int32, less_equal, where, fill, concat, reduce_sum, cd):
        train_op = mock.MagicMock(name='train_op', return_value='train_op')
        grad_norm = mock.MagicMock(name='grad_norm', return_value='grad_norm')
        get_train_ops.return_value = (train_op, 2, grad_norm, 4)
        variable(0.0, dtype=fw.float32).assign_sub = mock.MagicMock(return_value='assign_sub')
        fw.FLAGS.child_num_layers = 4
        fw.FLAGS.child_num_branches = 6
        fw.FLAGS.child_out_filters = 24
        fw.FLAGS.controller_tanh_constant = 0.5
        fw.FLAGS.controller_search_whole_channels = False
        with tf.Graph().as_default() as graph:
            variable(0.0, dtype=fw.float32)._as_graph_element = mock.MagicMock(return_value=graph)
            mc = MacroController(temperature=0.9)
            child_model = mock.MagicMock()
            child_model.build_valid_rl = mock.MagicMock(
                return_value=(
                    mock.MagicMock(return_value=('x_valid_shuffle', 'y_valid_shuffle')),
                    mock.MagicMock(return_value='vrl')))
            mc.skip_penaltys = 1.0
            self.assertEqual(('train_op', 2, 'grad_norm', 4), mc.build_trainer(child_model))
            child_model.build_valid_rl.assert_called_with()
            variable.assert_called_with(0, dtype=tf.int32, name='train_step')
            variable(0.0, dtype=fw.float32).assign_sub.assert_called_with(variable().__sub__().__rmul__())
            print.assert_any_call("-" * 80)
            print.assert_any_call("Building ConvController")
            print.assert_any_call("Build controller sampler")
            zeros.assert_called_with([1, 32], tf.float32)
            variable().assign_sub.assert_called_with(variable().__sub__().__rmul__())
            get_train_ops.assert_called_with(variable(), mc.learning_rate, clip_mode=mc.clip_mode, l2_reg=0.0, optim_algo=mc.optim_algo)
            train_op.assert_called_with(mc.loss, [])
            grad_norm.assert_called_with(mc.loss, [])
            to_float.assert_any_call(4.0)
            to_float.assert_called_with(6.0)
            reshape.assert_called_with(2.0, [-1])
            matmul.assert_called_with(2.0, variable())
            sscewl.assert_called_with(logits=0.0, labels=3.0)
            embedding_lookup.assert_called_with(variable(), 3.0)
            stack_lstm.assert_called_with('embedding_lookup', [1.0], [2.0], [variable(), variable()])
            multinomial.assert_called_with(0.0, 1)
            to_int32.assert_called_with('multinomial')
            where.assert_called_with('less_equal', x=0.0, y='fill')
            concat.assert_called_with([
                3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                3.0,
                3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                3.0, 4.0, 3.0, 4.0, 3.0,
                3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
                3.0, 4.0, 3.0, 4.0, 3.0], axis=0)
            reduce_sum.assert_called_with(4.0)
            cd.assert_called_with(['assign_sub'])

if "__main__" == __name__:
    unittest.main()
