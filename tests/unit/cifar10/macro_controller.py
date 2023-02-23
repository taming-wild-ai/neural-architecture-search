import unittest
import unittest.mock as mock
from unittest.mock import patch

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import src.framework as fw

from src.cifar10.macro_controller import MacroController
from src.cifar10.macro_child import DEFINE_integer # for child_num_layers, child_num_branches, child_out_filters

class TestMacroController(unittest.TestCase):
    @patch('src.cifar10.macro_controller.fw.to_float', return_value=tf.constant(np.ones((1, 2))))
    @patch('src.cifar10.macro_controller.fw.reduce_sum', return_value=4.0)
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
    @patch('src.cifar10.macro_controller.fw.get_variable', return_value="get_variable")
    @patch('src.cifar10.macro_controller.fw.zeros', return_value="zeros")
    @patch('src.cifar10.macro_controller.print')
    def test_constructor_not_whole_channels(self, print, zeros, get_variable, matmul, stack_lstm, multinomial, to_int32, reshape, sscewl, stop_gradient, embedding_lookup, concat, tanh, reduce_sum, to_float):
        with tf.Graph().as_default():
            mc = MacroController(temperature=0.9)
        self.assertEqual(MacroController, type(mc))
        print.assert_any_call('-' * 80)
        zeros.assert_called_with([1, 32], tf.float32)
        get_variable.assert_called_with('v', [32, 1])
        matmul.assert_called_with(2.0, 'get_variable')
        stack_lstm.assert_called_with('embedding_lookup', [1.0], [2.0], ['get_variable', 'get_variable'])
        multinomial.assert_called_with(0.5, 1)
        to_int32.assert_called_with('multinomial')
        reshape.assert_called_with(2.0, [-1])
        sscewl.assert_called_with(logits=0.5, labels=3.0)
        stop_gradient.assert_called_with(4.0)
        embedding_lookup.assert_called_with('get_variable', 3.0)
        concat.assert_called()
        tanh.assert_called()
        reduce_sum.assert_called()
        to_float.assert_called_with(3.0)

    @patch('src.cifar10.macro_controller.fw.to_float', return_value=tf.constant(np.ones((1, 2))))
    @patch('src.cifar10.macro_controller.fw.reduce_sum', return_value=4.0)
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
    @patch('src.cifar10.macro_controller.fw.get_variable', return_value="get_variable")
    @patch('src.cifar10.macro_controller.fw.zeros', return_value="zeros")
    @patch('src.cifar10.macro_controller.print')
    def test_constructor_whole_channels(self, print, zeros, get_variable, matmul, stack_lstm, multinomial, to_int32, reshape, sscewl, stop_gradient, embedding_lookup, concat, tanh, reduce_sum, to_float):
        fw.FLAGS.controller_search_whole_channels = True
        fw.FLAGS.child_num_layers = 4
        fw.FLAGS.child_num_branches = 6
        fw.FLAGS.child_out_filters = 24
        fw.FLAGS.controller_tanh_constant = 0.5
        with tf.Graph().as_default():
            self.assertEqual(MacroController, type(MacroController(temperature=0.9)))
        print.assert_any_call('-' * 80)
        zeros.assert_called_with([1, 32], tf.float32)
        get_variable.assert_called_with('v', [32, 1])
        matmul.assert_called_with(2.0, 'get_variable')
        stack_lstm.assert_called_with('embedding_lookup', [1.0], [2.0], ['get_variable', 'get_variable'])
        multinomial.assert_called_with(0.5, 1)
        to_int32.assert_called_with('multinomial')
        reshape.assert_called_with(2.0, [-1])
        sscewl.assert_called_with(logits=0.5, labels=3.0)
        stop_gradient.assert_called_with(4.0)
        embedding_lookup.assert_called_with('get_variable', 3.0)
        concat.assert_called_with([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], axis=0)
        tanh.assert_called()
        reduce_sum.assert_called()
        to_float.assert_called_with(3.0)

    @patch('src.cifar10.macro_controller.fw.control_dependencies')
    @patch('src.cifar10.macro_controller.fw.reduce_sum', return_value=4.0)
    @patch('src.cifar10.macro_controller.fw.concat', return_value=2.0)
    @patch('src.cifar10.macro_controller.fw.where', return_value="where")
    @patch('src.cifar10.macro_controller.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.macro_controller.fw.multinomial', return_value="multinomial")
    @patch('src.cifar10.macro_controller.fw.matmul', return_value=np.ones((5)))
    @patch('src.cifar10.macro_controller.stack_lstm', return_value=([1.0], [2.0]))
    @patch('src.cifar10.macro_controller.fw.embedding_lookup', return_value="embedding_lookup")
    @patch('src.cifar10.macro_controller.fw.sparse_softmax_cross_entropy_with_logits', return_value=0.5)
    @patch('src.cifar10.macro_controller.fw.reshape', return_value=3.0)
    @patch('src.cifar10.macro_controller.fw.to_float', return_value=tf.constant(np.ones((1, 2))))
    @patch('src.cifar10.macro_controller.get_train_ops', return_value=(1, 2, 3, 4))
    @patch('src.cifar10.macro_controller.fw.assign_sub', return_value="assign_sub")
    @patch('src.cifar10.macro_controller.fw.Variable', return_value=0.0)
    @patch('src.cifar10.macro_controller.fw.zeros', return_value="zeros")
    @patch('src.cifar10.macro_controller.print')
    def test_build_trainer(self, print, zeros, variable, assign_sub, get_train_ops, to_float, reshape, sscewl, embedding_lookup, stack_lstm, matmul, multinomial, to_int32, where, concat, reduce_sum, cd):
        fw.FLAGS.child_num_layers = 4
        fw.FLAGS.child_num_branches = 6
        fw.FLAGS.child_out_filters = 24
        fw.FLAGS.controller_tanh_constant = 0.5
        fw.FLAGS.controller_search_whole_channels = False
        with tf.Graph().as_default():
            mc = MacroController(temperature=0.9)
        child_model = mock.MagicMock()
        mc.skip_penaltys = 1.0
        mc.build_trainer(child_model)
        child_model.build_valid_rl.assert_called_with()
        assign_sub.assert_called()
        get_train_ops.assert_called()
        print.assert_any_call("-" * 80)
        print.assert_any_call("Building ConvController")
        print.assert_any_call("Build controller sampler")
        zeros.assert_called_with([1, 32], tf.float32)
        variable.assert_called_with(0, dtype=tf.int32, name='train_step')
        assign_sub.assert_called()
        get_train_ops.assert_called()
        to_float.assert_any_call(4.0)
        to_float.assert_called_with(6.0)
        reshape.assert_called_with(2.0, [-1])
        sscewl.assert_called()
        embedding_lookup.assert_called()
        stack_lstm.assert_called()
        matmul.assert_called()
        multinomial.assert_called()
        to_int32.assert_called_with('multinomial')
        where.assert_called()
        concat.assert_called()
        reduce_sum.assert_called_with(4.0)
        cd.assert_called_with(['assign_sub'])

if "__main__" == __name__:
    unittest.main()
