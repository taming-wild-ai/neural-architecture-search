import sys
import unittest
import unittest.mock as mock
from unittest.mock import patch

import numpy as np
from absl import flags
flags.FLAGS(['test'])
import src.framework as fw

from src.cifar10.macro_controller import MacroController
from src.cifar10.macro_child import DEFINE_integer # for child_num_layers, child_num_branches, child_out_filters

import tensorflow as tf

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
        mc = MacroController(temperature=0.9)
        logits, branch_ids = mc.sampler_logit()
        mc.sample_arc(branch_ids)
        log_prob, log_probs = mc.sample_log_prob(logits, branch_ids)
        mc.sample_entropy(log_probs)
        mc.skip_count(branch_ids)
        mc.skip_penaltys(logits)
        self.assertEqual(MacroController, type(mc))
        print.assert_any_call('-' * 80)
        zeros.assert_called_with([1, 32], tf.float32)
        rui().assert_called_with([32, 1])
        get_variable.assert_called_with(rui()(), name='v', trainable=True)
        matmul.assert_called_with(2.0, 'get_variable')
        stack_lstm.assert_called_with('embedding_lookup', [1.0], [2.0], ['get_variable', 'get_variable'])
        found = False
        for call in multinomial.call_args_list: # multinomial.assert_any_call(0.5, 1)
            if 2 == len(call[0]) and 0.5 == call[0][0] and 1 == call[0][1]:
                found = True
                break
        self.assertTrue(found)
        to_int32.assert_called_with('multinomial')
        # reshape.assert_any_call(2.0, [-1])
        found = False
        for call in reshape.call_args_list:
            if (type(call[0][0]) == float and type(call[0][1]) == list and 2.0 == call[0][0] and [-1] == call[0][1]):
                found = True
                break
        self.assertTrue(found)
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
        flags.FLAGS.controller_search_whole_channels = True
        flags.FLAGS.child_num_layers = 4
        flags.FLAGS.child_num_branches = 6
        flags.FLAGS.child_out_filters = 24
        flags.FLAGS.controller_tanh_constant = 0.5
        mc = MacroController(temperature=0.9)
        self.assertEqual(MacroController, type(mc))
        logits, branch_ids = mc.sampler_logit()
        mc.sample_arc(branch_ids)
        log_prob, log_probs = mc.sample_log_prob(logits, branch_ids)
        mc.sample_entropy(log_probs)
        mc.skip_count(branch_ids)
        mc.skip_penaltys(logits)
        print.assert_any_call('-' * 80)
        zeros.assert_called_with([1, 32], tf.float32)
        rui().assert_called_with([32, 1])
        get_variable.assert_called_with(rui()(), name='v', trainable=True)
        matmul.assert_called_with(2.0, 'get_variable')
        stack_lstm.assert_called_with('embedding_lookup', [1.0], [2.0], ['get_variable', 'get_variable'])
        multinomial.assert_called_with(0.5, 1)
        to_int32.assert_called_with('multinomial')
        # reshape.assert_any_call(2.0, [-1])
        found = False
        for call in reshape.call_args_list:
            found = (type(call[0][0]) == float and type(call[0][1]) == list and 2.0 == call[0][0] and [-1] == call[0][1])
            if found:
                break
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
    @patch('src.cifar10.macro_controller.fw.where', return_value=3.14)
    @patch('src.cifar10.macro_controller.fw.less_equal', return_value="less_equal")
    @patch('src.cifar10.macro_controller.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.macro_controller.fw.multinomial', return_value="multinomial")
    @patch('src.cifar10.macro_controller.fw.tanh', return_value=0.0)
    @patch('src.cifar10.macro_controller.fw.matmul')
    @patch('src.cifar10.macro_controller.stack_lstm', return_value=([1.0], [2.0]))
    @patch('src.cifar10.macro_controller.fw.embedding_lookup', return_value="embedding_lookup")
    @patch('src.cifar10.macro_controller.fw.sparse_softmax_cross_entropy_with_logits', return_value=0.5)
    @patch('src.cifar10.macro_controller.fw.reshape', return_value=3.0)
    @patch('src.cifar10.macro_controller.get_train_ops')
    @patch('src.cifar10.macro_controller.fw.zeros', return_value="zeros")
    @patch('src.cifar10.macro_controller.print')
    def test_build_trainer(self, print, zeros, get_train_ops, reshape, sscewl, embedding_lookup, stack_lstm, matmul, tanh, multinomial, to_int32, less_equal, where, fill, concat, reduce_sum, cd):
        train_op = mock.MagicMock(name='train_op', return_value='train_op')
        grad_norm = mock.MagicMock(name='grad_norm', return_value='grad_norm')
        get_train_ops.return_value = (train_op, 2, grad_norm, 4)
        # variable(0.0, dtype=fw.float32).assign_sub = mock.MagicMock(return_value='assign_sub')
        flags.FLAGS.child_num_layers = 4
        flags.FLAGS.child_num_branches = 6
        flags.FLAGS.child_out_filters = 24
        flags.FLAGS.controller_tanh_constant = 0.5
        flags.FLAGS.controller_search_whole_channels = False
        # variable(0.0, dtype=fw.float32)._as_graph_element = mock.MagicMock(return_value=tf.Graph())
        mc = MacroController(temperature=0.9)
        controller_logits, branch_ids = mc.sampler_logit()
        mc.sample_arc(branch_ids)
        log_prob, log_probs = mc.sample_log_prob(controller_logits, branch_ids)
        mc.sample_entropy(log_probs)
        mc.skip_count(branch_ids)
        mc.skip_penaltys(controller_logits)
        child_model = mock.MagicMock()
        child_model.batch_size = 6
        shuffle = mock.MagicMock(return_value=('x_valid_shuffle', 'y_valid_shuffle'))
        vrl = mock.MagicMock(return_value=42)
        mc.skip_penaltys = 1.0
        self.assertEqual((train_op, 2, grad_norm, 4), mc.build_trainer(child_model, vrl))
        mc.skip_rate(branch_ids)
        mc.sample_log_prob(controller_logits, branch_ids)
        train_op(mc.loss, [])
        grad_norm(mc.loss, [])
        mc.loss('child_logits', 'y_valid_shuffle', controller_logits, branch_ids, log_probs)
        # variable.assert_called_with(0, dtype=tf.int32, name='train_step')
        # variable(0.0, dtype=fw.float32).assign_sub.assert_called_with(variable().__sub__().__rmul__())
        print.assert_any_call("-" * 80)
        print.assert_any_call("Building ConvController")
        print.assert_any_call("Build controller sampler")
        zeros.assert_called_with([1, 32], tf.float32)
        # variable().assign_sub.assert_called_with(variable().__sub__().__rmul__())
        # get_train_ops.assert_called_with(variable(), mc.learning_rate, clip_mode=mc.clip_mode, l2_reg=0.0, optim_algo=mc.optim_algo)
        train_op.assert_called_with(mc.loss, [])
        grad_norm.assert_called_with(mc.loss, [])
        # to_float.assert_any_call(4.0)
        # to_float.assert_any_call(6.0)
        # to_float.assert_called_with(child_model.batch_size)
        found = False
        for call in reshape.call_args_list: # reshape.assert_any_call(2.0, [-1])
            if 2 == len(call[0]):
                if float == type(call[0][0]) and list == type(call[0][1]) and 2.0 == call[0][0] and [-1] == call[0][-1]:
                    found = True
                    break
        self.assertTrue(found)
        # matmul.assert_called_with(2.0, variable())
        sscewl.assert_called_with(logits=0.0, labels=3.0)
        # embedding_lookup.assert_called_with(variable(), 3.0)
        # stack_lstm.assert_called_with('embedding_lookup', [1.0], [2.0], [variable(), variable()])
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
        found = False
        for call in reduce_sum.call_args_list:         # reduce_sum.assert_any_call(4.0)
            if 1 == len(call[0]):
                if float == type(call[0][0]) and 4.0 == call[0][0]:
                    found = True
                    break
        self.assertTrue(found)
        # cd.assert_called_with(['assign_sub'])

if "__main__" == __name__:
    unittest.main()
