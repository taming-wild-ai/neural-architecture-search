import unittest
import unittest.mock as mock
from unittest.mock import patch

import numpy as np
from absl import flags
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import src.framework as fw

from src.cifar10.child import DataFormat, Child
from src.cifar10.macro_child import MacroChild
from src.cifar10.macro_controller import DEFINE_boolean # for controller_search_whole_channels

def mock_init(self, images, labels, **kwargs):
    self.whole_channels = False
    self.data_format = DataFormat.new("NCHW")
    self.cutout_size = None
    self.num_layers = 2
    self.use_aux_heads = False
    self.num_train_batches = 1
    self.fixed_arc = None
    self.out_filters = 24
    self.learning_rate = mock.MagicMock()
    self.weights = fw.WeightRegistry()
    self.x_train, self.y_train = None, None
    self.x_valid, self.y_valid = None, None
    self.x_test, self.y_test = None, None

def mock_init_nhwc(self, images, labels, **kwargs):
    self.whole_channels = False
    self.data_format = DataFormat.new("NHWC")
    self.cutout_size = None
    self.num_layers = 2
    self.use_aux_heads = False
    self.num_train_batches = 1
    self.fixed_arc = None
    self.out_filters = 24
    self.learning_rate = mock.MagicMock()
    self.weights =fw.WeightRegistry()
    self.x_train, self.y_train = None, None
    self.x_valid, self.y_valid = None, None
    self.x_test, self.y_test = None, None

def mock_init_invalid(self, images, labels, **kwargs):
    self.whole_channels = False
    self.data_format = DataFormat.new("INVALID")
    self.weights = fw.WeightRegistry()


class TestMacroChild(unittest.TestCase):
    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    def test_init(self):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        self.assertEqual(MacroChild, type(mc))

    @patch('src.cifar10.child.Child.__init__', new=mock_init_invalid)
    def test_get_strides_exception(self):
        with tf.Graph().as_default():
            self.assertRaises(KeyError, MacroChild, {}, {})

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    def test_factorized_reduction_failure(self):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            self.assertRaises(AssertionError, Child.FactorizedReduction, mc, 3, 3, None, None, mc.weights, False)

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.conv2d', return_value='conv2d')
    @patch('src.cifar10.child.BatchNorm')
    def test_factorized_reduction_stride1(self, batch_norm, conv2d):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            self.assertRaises(AssertionError, Child.FactorizedReduction, mc, 3, 3, None, None, mc.weights, False)
            with patch.object(mc.weights, 'get') as create_weight:
                fr = Child.FactorizedReduction(mc, None, 2, 1, True, mc.weights, False)
                _ = fr(None)
                create_weight.assert_called_with(False, 'path_conv/', "w", [1, 1, None, 2], None)
                batch_norm.assert_called_with(True, mc.data_format, mc.weights, 2, False)
                batch_norm().assert_called_with('conv2d')

    zeros = np.zeros((5, 5, 5, 5))
    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.fw.avg_pool', return_value="path1")
    @patch('src.cifar10.macro_child.fw.pad')
    @patch('src.cifar10.macro_child.fw.concat', return_value="final_path")
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="final_path"))
    def test_factorized_reduction_nhwc(self, batch_norm, concat, pad, avg_pool, conv2d):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                fr = Child.FactorizedReduction(mc, 'inp_c', 2, 2, True, mc.weights, False)
                self.assertEqual("final_path", fr(None))
                avg_pool.assert_any_call(None, [1, 1, 1, 1], [1, 2, 2, 1], 'VALID', data_format='NHWC')
                create_weight.assert_any_call(False, 'path1_conv/', "w", [1, 1, 'inp_c', 1], None)
                pad.assert_called_with(None, [[0, 0], [0, 1], [0, 1], [0, 0]])
                avg_pool.assert_called_with(pad().__getitem__(), [1, 1, 1, 1], [1, 2, 2, 1], 'VALID', data_format='NHWC')
                create_weight.assert_called_with(False, 'path2_conv/', "w", [1, 1, 'inp_c', 1], None)
                conv2d.assert_called_with('path1', 'fw.create_weight', [1, 1, 1, 1], 'SAME', data_format='NHWC')
                concat.assert_called_with(values=["conv2d", "conv2d"], axis=3)
                batch_norm.assert_called_with(True, mc.data_format, mc.weights, 2, False)
                batch_norm().assert_called_with('final_path')

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.fw.avg_pool', return_value="path1")
    @patch('src.cifar10.macro_child.fw.pad')
    @patch('src.cifar10.macro_child.fw.concat', return_value="final_path")
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="final_path"))
    def test_factorized_reduction_nchw(self, batch_norm, concat, pad, avg_pool, conv2d):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                fr = Child.FactorizedReduction(mc, 'inp_c', 2, 2, True, mc.weights, False)
                self.assertEqual("final_path", fr(None))
                avg_pool.assert_any_call(None, [1, 1, 1, 1], [1, 1, 2, 2], 'VALID', data_format=mc.data_format.name)
                create_weight.assert_any_call(False, 'path1_conv/', "w", [1, 1, 'inp_c', 1], None)
                pad.assert_called_with(None, [[0, 0], [0, 0], [0, 1], [0, 1]])
                avg_pool.assert_called_with(pad().__getitem__(), [1, 1, 1, 1], [1, 1, 2, 2], 'VALID', data_format='NCHW')
                create_weight.assert_called_with(False, 'path2_conv/', "w", [1, 1, 'inp_c', 1], None)
                conv2d.assert_called_with('path1', 'fw.create_weight', [1, 1, 1, 1], 'SAME', data_format='NCHW')
                concat.assert_called_with(values=["conv2d", "conv2d"], axis=1)
                batch_norm.assert_called_with(True, mc.data_format, mc.weights, 2, False)
                batch_norm().assert_called_with('final_path')

    @patch('src.cifar10.child.Child.__init__', new=mock_init_invalid)
    def test_data_format_raises(self):
        with tf.Graph().as_default():
            self.assertRaises(KeyError, MacroChild, {}, {})

    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    @patch('src.cifar10.macro_child.MacroChild.ENASLayer', return_value=mock.MagicMock(return_value='enas_layer'))
    @patch('src.cifar10.child.Child.InputConv', return_value='input_conv')
    @patch('src.cifar10.macro_child.print')
    @patch('src.cifar10.macro_child.fw.matmul', return_value="matmul")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="final_path"))
    @patch('src.cifar10.macro_child.fw.dropout')
    def test_model_nhwc(self, dropout, batch_norm, conv2d, matmul, _print, input_conv, enas_layer):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            mc.name = "generic_model"
            mc.keep_prob = 0.9
            with patch.object(mc.data_format, 'global_avg_pool', return_value='gap') as global_avg_pool:
                with patch.object(mc.weights, 'get', return_value='w') as create_weight:
                    model = MacroChild.Model(mc, True)
                    model({})
                    create_weight.assert_any_call(False, 'generic_model/stem_conv/', "w", [3, 3, 3, 24], None)
                    conv2d.assert_called_with({}, 'w', [1, 1, 1, 1], 'SAME', data_format='NHWC')
                    enas_layer.assert_called_with(mc, 1, 18, 24, 24, True, mc.weights, False, 'input_conv', 'input_conv')
                    enas_layer().assert_called_with(['final_path', 'enas_layer', 'enas_layer'])
                    global_avg_pool.assert_called_with('enas_layer')
                    dropout.assert_called_with('gap', 0.9)
                    create_weight.assert_called_with(False, 'generic_model/fc/', "w", [24, 10], None)
                    matmul.assert_called_with(dropout(), "w")

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.MacroChild.ENASLayer', return_value=mock.MagicMock(return_value='enas_layer'))
    @patch('src.cifar10.child.Child.FactorizedReduction')
    @patch('src.cifar10.child.Child.InputConv', return_value='input_conv')
    @patch('src.cifar10.macro_child.print')
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="final_path"))
    @patch('src.cifar10.macro_child.fw.dropout', return_value=tf.constant(np.ndarray((4, 3, 32, 32))))
    @patch('src.cifar10.macro_child.fw.matmul')
    def test_model_nchw_no_whole_channels(self, matmul, dropout, batch_norm, conv2d, _print, input_conv, fr, enas_layer):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            mc.name = "generic_model"
            mc.keep_prob = 0.9
            mc.pool_layers = [0]
            with patch.object(mc.data_format, 'global_avg_pool', return_value="global_avg_pool") as global_avg_pool:
                with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                    model = MacroChild.Model(mc, True)
                    model({})
                    create_weight.assert_any_call(False, 'generic_model/stem_conv/', "w", [3, 3, 3, 24], None)
                    conv2d.assert_called_with({}, 'fw.create_weight', [1, 1, 1, 1], 'SAME', data_format='NCHW')
                    batch_norm.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                    batch_norm().assert_called_with('conv2d')
                    fr.assert_called_with(mc, 24, 24, 2, True, mc.weights, False)
                    fr().assert_called_with('enas_layer')
                    enas_layer.assert_called_with(mc, 1, 18, 24, 24, True, mc.weights, False, 'input_conv', 'input_conv')
                    enas_layer().assert_called_with([fr()(), fr()(), 'enas_layer'])
                    global_avg_pool.assert_called_with('enas_layer')
                    dropout.assert_called_with('global_avg_pool', 0.9)
                    create_weight.assert_called_with(False, 'generic_model/fc/', "w", [24, 10], None)
                    matmul.assert_called_with(dropout.return_value, 'fw.create_weight')

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.MacroChild.FixedLayer', return_value=mock.MagicMock(return_value='fixed_layer'))
    @patch('src.cifar10.macro_child.print')
    @patch('src.cifar10.child.Child.FactorizedReduction')
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="final_path"))
    @patch('src.cifar10.macro_child.fw.dropout', return_value=tf.constant(np.ndarray((4, 3, 32, 32))))
    @patch('src.cifar10.macro_child.fw.matmul')
    def test_model_nchw_whole_channels(self, matmul, dropout, batch_norm, conv2d, fr, _print, fixed_layer):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            mc.name = "generic_model"
            mc.keep_prob = 0.9
            mc.pool_layers = [0]
            mc.whole_channels = True
            mc.fixed_arc = ""
            with patch.object(mc.data_format, 'global_avg_pool', return_value='global_avg_pool') as global_avg_pool:
                with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                    model = MacroChild.Model(mc, True)
                    model({})
                    create_weight.assert_any_call(False, 'generic_model/stem_conv/', "w", [3, 3, 3, 24], None)
                    conv2d.assert_called_with({}, 'fw.create_weight', [1, 1, 1, 1], 'SAME', data_format='NCHW')
                    batch_norm.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                    batch_norm().assert_called_with('conv2d')
                    fr.assert_called_with(mc, 24, 48, 2, True, mc.weights, False)
                    fr().assert_called_with('fixed_layer')
                    fixed_layer.assert_called_with(mc, 1, 1, 48, 48, True, mc.weights, False)
                    fixed_layer().assert_called_with([fr()(), fr()(), 'fixed_layer'])
                    global_avg_pool.assert_called_with('fixed_layer')
                    dropout.assert_called_with('global_avg_pool', 0.9)
                    create_weight.assert_called_with(False, 'generic_model/fc/', "w", [48, 10], None)
                    matmul.assert_called_with(dropout.return_value, 'fw.create_weight')

    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    @patch('src.cifar10.macro_child.MacroChild.PoolBranch')
    @patch('src.cifar10.macro_child.MacroChild.ConvBranch')
    @patch('src.cifar10.macro_child.fw.case', return_value=tf.constant(np.ndarray((4, 32, 32, 24)), tf.float32))
    @patch('src.cifar10.macro_child.fw.add_n', return_value="add_n")
    @patch('src.cifar10.macro_child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    def test_enas_layer_whole_channels_nhwc(self, batch_norm, add_n, case, conv_branch, pool_branch):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            mc.whole_channels = True
            mc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0".split(" ") if x])
            with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
                el = MacroChild.ENASLayer(mc, 1, 0, 3, 24, True, mc.weights, False, None, None)
                el([input_tensor])
                conv_branch.assert_called_with(mc, 5, True, 24, 3, 24, mc.weights, False, 1, 0, True)
                conv_branch().assert_called_with(input_tensor)
                case.assert_called()
                batch_norm.assert_called_with(True, mc.data_format, mc.weights, 3, False)
                batch_norm().assert_called_with('add_n')
                create_weight.assert_not_called()
                pool_branch.assert_called_with(mc, 24, 'max', None, 0)
                pool_branch().assert_called_with(input_tensor)

    @patch('src.cifar10.macro_child.MacroChild.PoolBranch')
    @patch('src.cifar10.macro_child.MacroChild.ConvBranch')
    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.case', return_value=tf.constant(np.ndarray((4, 24, 32, 32)), tf.float32))
    @patch('src.cifar10.macro_child.fw.add_n', return_value="add_n")
    @patch('src.cifar10.macro_child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    def test_enas_layer_whole_channels_nchw(self, batch_norm, add_n, case, conv_branch, pool_branch):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.whole_channels = True
        mc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0".split(" ") if x])
        with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
            input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
            el = MacroChild.ENASLayer(mc, 1, 0, 3, 24, True, mc.weights, False, None, None)
            el([input_tensor])
            conv_branch.assert_called_with(mc, 5, True, 24, 3, 24, mc.weights, False, 1, 0, True)
            conv_branch().assert_called_with(input_tensor)
            case.assert_called()
            batch_norm.assert_called_with(True, mc.data_format, mc.weights, 3, False)
            batch_norm().assert_called_with('add_n')
            create_weight.assert_not_called() # Assumes called inside mocked PoolBranch or ConvBranch
            pool_branch.assert_called_with(mc, 24, 'max', None, 0)
            pool_branch().assert_called_with(input_tensor)

    @patch('src.cifar10.macro_child.MacroChild.PoolBranch')
    @patch('src.cifar10.macro_child.MacroChild.ConvBranch')
    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    @patch('src.cifar10.macro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.macro_child.fw.boolean_mask', return_value="boolean_mask")
    @patch('src.cifar10.macro_child.fw.logical_or', return_value="logical_or")
    @patch('src.cifar10.macro_child.fw.concat', return_value="concat")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.macro_child.fw.add_n', return_value="add_n")
    def test_enas_layer_not_whole_channels_nhwc(self, add_n, batch_norm, relu, conv_2d, concat, logical_or, boolean_mask, reshape, conv_branch, pool_branch):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            mc.whole_channels = False
            mc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0 2 0 0 1 2 0 0 0 0".split(" ") if x])
            with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
                el = MacroChild.ENASLayer(mc, 1, 0, 3, 24, True, mc.weights, False, None, None)
                el([input_tensor])
                conv_branch.assert_called_with(mc, 5, True, 0, 3, 24, mc.weights, False, 1, 2, True)
                conv_branch().assert_called_with(input_tensor)
                pool_branch.assert_called_with(mc, 0, 'max', None, 2)
                pool_branch().assert_called_with(input_tensor)
                concat.assert_called_with([conv_branch()(), conv_branch()(), conv_branch()(), conv_branch()(), pool_branch()(), pool_branch()()], axis=3)
                conv_2d.assert_called_with('concat', 'reshape', [1, 1, 1, 1], 'SAME', data_format='NHWC')
                relu.assert_called_with("batch_norm")
                batch_norm.assert_called_with(True, mc.data_format, mc.weights, 3, False)
                batch_norm().assert_called_with('add_n')
                create_weight.assert_any_call(False, 'final_conv/', 'w', [144, 24], None)
                boolean_mask.assert_called_with('fw.create_weight', 'logical_or')
                reshape.assert_called_with('boolean_mask', [1, 1, -1, 24])

    @patch('src.cifar10.macro_child.MacroChild.PoolBranch')
    @patch('src.cifar10.macro_child.MacroChild.ConvBranch')
    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.concat', return_value="concat")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.macro_child.fw.add_n', return_value="add_n")
    @patch('src.cifar10.macro_child.fw.logical_and', return_value="logical_and")
    @patch('src.cifar10.macro_child.fw.logical_or', return_value="logical_or")
    @patch('src.cifar10.macro_child.fw.boolean_mask', return_value="boolean_mask")
    @patch('src.cifar10.macro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.macro_child.fw.shape', return_value=["shape"])
    def test_enas_layer_not_whole_channels_nchw(self, shape, reshape, boolean_mask, logical_or, logical_and, add_n, batch_norm, relu, conv_2d, concat, conv_branch, pool_branch):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            mc.whole_channels = False
            mc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0 2 0 0 1 2 0 0 0 0".split(" ") if x])
            with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
                el = MacroChild.ENASLayer(mc, 1, 0, 3, 24, True, mc.weights, False, None, None)
                el([input_tensor])
                conv_branch.assert_called_with(mc, 5, True, 0, 3, 24, mc.weights, False, 1, 2, True)
                conv_branch().assert_called_with(input_tensor)
                pool_branch.assert_called_with(mc, 0, 'max', None, 2)
                pool_branch().assert_called_with(input_tensor)
                concat.assert_called_with([conv_branch()(), conv_branch()(), conv_branch()(), conv_branch()(), pool_branch()(), pool_branch()()], axis=1)
                conv_2d.assert_called_with('reshape', 'reshape', [1, 1, 1, 1], 'SAME', data_format='NCHW')
                relu.assert_called_with("batch_norm")
                batch_norm.assert_called_with(True, mc.data_format, mc.weights, 3, False)
                batch_norm().assert_called_with('add_n')
                logical_and.assert_called()
                logical_or.assert_called_with("logical_or", "logical_and")
                boolean_mask.assert_called_with("fw.create_weight", "logical_or")
                shape.assert_called_with(input_tensor)
                reshape.assert_any_call("concat", ['shape', -1, 32, 3])
                reshape.assert_called_with('boolean_mask', [1, 1, -1, 24])
                create_weight.assert_called_with(False, 'final_conv/', 'w', [144, 24], None)

    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    def test_fixed_layer_whole_channels_nhwc(self, batch_norm, conv2d, relu):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                mc.whole_channels = True
                mc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0".split(" ") if x])
                input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
                fl = MacroChild.FixedLayer(mc, 0, 0, 3, 24, True, mc.weights, False)
                fl([input_tensor])
                relu.assert_called_with("batch_norm")
                conv2d.assert_called_with('relu', 'fw.create_weight', [1, 1, 1, 1], 'SAME', data_format='NHWC')
                batch_norm.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                batch_norm().assert_called_with('conv2d')
                create_weight.assert_called_with(False, 'conv_3x3/', 'w', [3, 3, 24, 24], None)

    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.macro_child.fw.concat', return_value="concat")
    def test_fixed_layer_whole_channels_nhwc_second_layer(self, concat, batch_norm, conv2d, relu):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                mc.whole_channels = True
                mc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0".split(" ") if x])
                input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
                fl = MacroChild.FixedLayer(mc, 1, 0, 3, 24, True, mc.weights, False)
                fl([input_tensor])
                create_weight.assert_called_with(False, 'skip/', "w", [1, 1, 96, 24], None)
                relu.assert_called_with("concat")
                conv2d.assert_called_with('relu', 'fw.create_weight', [1, 1, 1, 1], 'SAME', data_format='NHWC')
                batch_norm.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                batch_norm().assert_called_with('conv2d')

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    def test_fixed_layer_whole_channels_nchw(self, batch_norm, conv2d, relu):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                mc.whole_channels = True
                mc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0".split(" ") if x])
                input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
                fl = MacroChild.FixedLayer(mc, 0, 0, 3, 24, True, mc.weights, False)
                fl([input_tensor])
                relu.assert_called_with("batch_norm")
                conv2d.assert_called_with('relu', 'fw.create_weight', [1, 1, 1, 1], 'SAME', data_format='NCHW')
                batch_norm.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                batch_norm().assert_called_with('conv2d')
                create_weight.assert_called_with(False, 'conv_3x3/', 'w', [3, 3, 24, 24], None)

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    def test_fixed_layer_whole_channels_nchw_raises(self):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.whole_channels = True
        mc.sample_arc = np.array([int(x) for x in "6 3 0 0 1 0".split(" ") if x])
        self.assertRaises(ValueError, MacroChild.FixedLayer, mc, 0, 0, 3, 24, True, mc.weights, False)

    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    @patch('src.cifar10.macro_child.MacroChild.PoolBranch')
    @patch('src.cifar10.macro_child.MacroChild.ConvBranch')
    @patch('src.cifar10.macro_child.Child.InputConv')
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.macro_child.fw.concat', return_value="concat")
    def test_fixed_layer_not_whole_channels_nhwc(self, concat, batch_norm, conv2d, relu, input_conv, conv_branch, pool_branch):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            mc.whole_channels = False
            mc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0 2 0 0 1 2 0 0 0 0".split(" ") if x])
            input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
            with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                fl = MacroChild.FixedLayer(mc, 0, 0, 3, 24, True, mc.weights, False)
                fl([input_tensor])
                conv_branch.assert_called_with(mc, 5, True, 0, 3, 24, mc.weights, False, 1, 0, True)
                conv_branch().assert_called_with(input_tensor)
                relu.assert_called_with("concat")
                conv2d.assert_called_with('relu', 'fw.create_weight', [1, 1, 1, 1], 'SAME', data_format='NHWC')
                batch_norm.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                batch_norm().assert_called_with('conv2d')
                pool_branch.assert_called_with(mc, 0, "max", input_conv())
                pool_branch().assert_called_with(input_tensor)
                concat.assert_called_with([conv_branch()(), conv_branch()(), conv_branch()(), conv_branch()(), pool_branch()(), pool_branch()()], axis=3)
                create_weight.assert_called_with(False, 'final_conv/', 'w', [1, 1, 4, 24], None)

    @patch('src.cifar10.macro_child.MacroChild.PoolBranch')
    @patch('src.cifar10.macro_child.MacroChild.ConvBranch')
    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.Child.InputConv')
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.macro_child.fw.concat', return_value="concat")
    def test_fixed_layer_not_whole_channels_nchw_second_layer(self, concat, batch_norm, conv2d, relu, input_conv, conv_branch, pool_branch):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            mc.whole_channels = False
            mc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0 2 0 0 1 2 0 0 0 0".split(" ") if x])
            input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
            with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                fl = MacroChild.FixedLayer(mc, 1, 0, 3, 24, True, mc.weights, False)
                fl([input_tensor])
                conv_branch.assert_called_with(mc, 5, True, 0, 3, 24, mc.weights, False, 1, 0, True)
                conv_branch().assert_called_with(input_tensor)
                relu.assert_called_with('concat')
                conv2d.assert_called_with('relu', 'fw.create_weight', [1, 1, 1, 1], 'SAME', data_format='NCHW')
                batch_norm.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                batch_norm().assert_called_with('conv2d')
                pool_branch.assert_called_with(mc, 0, "max", input_conv())
                pool_branch().assert_called_with(input_tensor)
                concat.assert_called_with(['batch_norm'], axis=1)
                create_weight.assert_called_with(False, 'skip/', 'w', [1, 1, 24, 24], None)

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    def test_conv_branch_failure(self):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        self.assertRaises(AssertionError, MacroChild.ConvBranch, mc, None, 3, 24, True, 0, 24, mc.weights, False, None, False)

    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value='batch_norm'))
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    def test_conv_branch_nhwc(self, relu, batch_norm1, batch_norm, conv2d):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                mc.fixed_arc = "0 3 0 0 1 0"
                input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
                cb = MacroChild.ConvBranch(mc, 24, True, 0, 3, 24, mc.weights, False, 1, None, False)
                cb(input_tensor)
                conv2d.assert_called_with('relu', 'fw.create_weight', [1, 1, 1, 1], 'SAME', data_format='NHWC')
                batch_norm.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                batch_norm().assert_called_with('conv2d')
                relu.assert_called_with("batch_norm")
                create_weight.assert_called_with(False, 'out_conv_24/', 'w', [24, 24, 3, 0], None)

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    def test_conv_branch_nchw(self, relu, batch_norm1, batch_norm, conv2d):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                mc.fixed_arc = "0 3 0 0 1 0"
                input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
                cb = MacroChild.ConvBranch(mc, 24, True, 0, 3, 24, mc.weights, False, 1, None, False)
                cb(input_tensor)
                create_weight.assert_any_call(False, 'inp_conv_1/', "w", [1, 1, 3, 24], None)
                conv2d.assert_called_with('relu', 'fw.create_weight', [1, 1, 1, 1], 'SAME', data_format='NCHW')
                batch_norm1.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                batch_norm1().assert_called_with('conv2d')
                relu.assert_called_with("batch_norm")
                create_weight.assert_called_with(False, 'out_conv_24/', "w", [24, 24, 3, 0], None)

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value='batch_norm'))
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.transpose', return_value="transpose")
    @patch('src.cifar10.macro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.macro_child.fw.separable_conv2d', return_value="sep_conv2d")
    @patch('src.cifar10.macro_child.BatchNormWithMask', return_value=mock.MagicMock(return_value="bnwm"))
    def test_conv_branch_nchw_separable(self, bnwm, sep_conv2d, reshape, transpose, relu, batch_norm1, batch_norm, conv2d):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                mc.fixed_arc = "0 3 0 0 1 0"
                mc.filter_size = 24
                input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
                cb = MacroChild.ConvBranch(mc, 24, True, 0, 3, 24, mc.weights, False, 1, None, True)
                cb(input_tensor)
                create_weight.assert_any_call(False, 'inp_conv_1/', "w", [1, 1, 3, 24], None)
                conv2d.assert_called_with(input_tensor, 'fw.create_weight', [1, 1, 1, 1], 'SAME', data_format='NCHW')
                batch_norm.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                batch_norm().assert_called_with('sep_conv2d')
                relu.assert_called_with("batch_norm")

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.transpose', return_value=tf.Variable(initial_value=np.zeros((24, 24, 24, 1))))
    @patch('src.cifar10.macro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.macro_child.fw.separable_conv2d', return_value="sep_conv2d")
    @patch('src.cifar10.macro_child.BatchNormWithMask', return_value=mock.MagicMock(return_value="bnwm"))
    def test_conv_branch_nchw_second_index(self, bnwm, sep_conv2d, reshape, transpose, relu, batch_norm1, batch_norm, conv2d):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                    mc.fixed_arc = "0 3 0 0 1 0"
                    input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
                    cb = MacroChild.ConvBranch(mc, 24, True, 0, 3, 24, mc.weights, False, 1, 1, False)
                    cb(input_tensor)
                    create_weight.assert_any_call(False, 'inp_conv_1/', "w", [1, 1, 3, 24], None)
                    conv2d.assert_called_with('relu', transpose.return_value, [1, 1, 1, 1], 'SAME', data_format='NCHW')
                    batch_norm1.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                    batch_norm1().assert_called_with('conv2d')
                    relu.assert_called_with("bnwm")
                    create_weight.assert_called_with(False, 'out_conv_24/', "w", [24, 24, 24, 24], None)

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.transpose', return_value="transpose")
    @patch('src.cifar10.macro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.macro_child.fw.separable_conv2d', return_value="sep_conv2d")
    @patch('src.cifar10.macro_child.BatchNormWithMask', return_value=mock.MagicMock(return_value="bnwm"))
    def test_conv_branch_nchw_second_index_separable(self, bnwm, sep_conv2d, reshape, transpose, relu, batch_norm1, batch_norm, conv2d):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            with patch.object(mc.weights, 'get', return_value=tf.Variable(initial_value=np.zeros((24, 24, 24, 1)))) as create_weight:
                mc.fixed_arc = "0 3 0 0 1 0"
                input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
                cb = MacroChild.ConvBranch(mc, 24, True, 0, 3, 24, mc.weights, False, 1, 1, True)
                cb(input_tensor)
                create_weight.assert_any_call(False, 'inp_conv_1/', "w", [1, 1, 3, 24], None)
                conv2d.assert_called_with(input_tensor, mc.weights.get.return_value, [1, 1, 1, 1], 'SAME', data_format='NCHW')
                batch_norm1.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                batch_norm1().assert_called_with('conv2d')
                relu.assert_called_with("bnwm")
                create_weight.assert_called_with(False, 'out_conv_24/', "w_point", [24, 24], None)

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    def test_pool_branch_failure(self):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            with fw.name_scope("conv_1") as scope:
                input_conv = MacroChild.InputConv(
                    mc.weights,
                    False,
                    scope,
                    1,
                    3,
                    mc.out_filters,
                    True,
                    mc.data_format)
        self.assertRaises(AssertionError, MacroChild.PoolBranch, mc, None, None, input_conv, None)

    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.avg_pool2d')
    def test_pool_branch_nhwc_avg(self, avg_pool2d, relu, batch_norm1, batch_norm, conv2d):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            with patch.object(mc.weights, 'get', return_value=tf.Variable(initial_value=np.zeros((24, 24, 24, 1)))) as create_weight:
                mc.fixed_arc = "0 3 0 0 1 0"
                input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
                with fw.name_scope("conv_1") as scope:
                    input_conv = MacroChild.InputConv(
                        mc.weights,
                        False,
                        scope,
                        1,
                        3,
                        mc.out_filters,
                        True,
                        mc.data_format)
                pb = MacroChild.PoolBranch(mc, 0, "avg", input_conv, True)
                pb(input_tensor)
                conv2d.assert_called_with(input_tensor, mc.weights.get.return_value, [1, 1, 1, 1], 'SAME', data_format='NHWC')
                batch_norm1.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                batch_norm1().assert_called_with('conv2d')
                relu.assert_called_with("batch_norm")
                avg_pool2d.assert_called_with([3, 3], [1, 1], "SAME", data_format="channels_last")
                avg_pool2d().assert_called_with('relu')
                create_weight.assert_called_with(False, 'conv_1/', 'w', [1, 1, 3, 24], None)

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.BatchNorm', return_value="batch_norm")
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.max_pool2d')
    def test_pool_branch_nchw_max(self, max_pool2d, relu, batch_norm1, batch_norm, conv2d):
        flags.FLAGS.controller_search_whole_channels = True
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            with patch.object(mc.weights, 'get', return_value=tf.Variable(initial_value=np.zeros((24, 24, 24, 1)))) as create_weight:
                mc.fixed_arc = "0 3 0 0 1 0"
                input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
                with fw.name_scope("conv_1") as scope:
                    input_conv = Child.InputConv(
                        mc.weights,
                        False,
                        scope,
                        1,
                        3,
                        mc.out_filters,
                        True,
                        mc.data_format)
                pb = MacroChild.PoolBranch(mc, 0, "max", input_conv, True)
                pb(input_tensor)
                conv2d.assert_called_with(input_tensor, mc.weights.get.return_value, [1, 1, 1, 1], 'SAME', data_format='NCHW')
                batch_norm1.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                batch_norm1().assert_called_with('conv2d')
                relu.assert_called_with("batch_norm")
                max_pool2d.assert_called_with([3, 3], [1, 1], 'SAME', data_format='channels_first')
                max_pool2d().assert_called_with('relu')
                create_weight.assert_called_with(False, 'conv_1/', 'w', [1, 1, 3, 24], None)

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.MacroChild.Model', return_value=mock.MagicMock(return_value='model'))
    @patch('src.cifar10.macro_child.fw.get_or_create_global_step', return_value='global_step')
    @patch('src.cifar10.macro_child.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.macro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.macro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.macro_child.print')
    @patch('src.cifar10.macro_child.fw.sparse_softmax_cross_entropy_with_logits', return_value="sscewl")
    @patch('src.cifar10.macro_child.fw.reduce_mean', return_value="reduce_mean")
    @patch('src.cifar10.macro_child.fw.argmax', return_value=10)
    @patch('src.cifar10.macro_child.get_train_ops')
    def test_build_train(self, get_train_ops, argmax, reduce_mean, sscewl, print1, to_int32, equal, reduce_sum, global_step, model):
        train_op = mock.MagicMock(name='train_op', return_value='train_op')
        grad_norm = mock.MagicMock(name='grad_norm', return_value='grad_norm')
        get_train_ops.return_value = (train_op, 2, grad_norm, 4)
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
            mc.x_train = 1
            mc.y_train = 2
            mc.clip_mode = None
            mc.grad_bound = None
            mc.l2_reg = 1e-4
            mc.lr_init = 0.1
            mc.lr_dec_start = 0
            mc.lr_dec_every=100
            mc.lr_dec_rate = 0.1
            mc.num_train_batches = 310
            mc.optim_algo = None
            mc.sync_replicas = False
            mc.num_aggregate = None
            mc.num_replicas = None
            mc.name = "macro_child"
            loss0, train_loss0, train_acc0, global_step0, train_op0, lr, grad_norm0, optimizer = mc._build_train(mc.y_train)
            logits = MacroChild.Model(mc, True)(mc.x_train)
            loss = loss0(logits)
            train_acc = train_acc0(logits)
            train_op = train_op0(loss, mc.tf_variables())
            grad_norm = grad_norm0(loss, mc.tf_variables())
            self.assertEqual(loss, 'reduce_mean')
            self.assertEqual('reduce_sum', train_acc)
            self.assertEqual('train_op', train_op)
            self.assertEqual(2, lr)
            self.assertEqual('grad_norm', grad_norm)
            self.assertEqual(4, optimizer)
            print1.assert_any_call("-" * 80)
            print1.assert_any_call("Build train graph")
            model.assert_called_with(mc, True)
            model().assert_called_with(mc.x_train)
            sscewl.assert_called_with(logits="model", labels=mc.y_train)
            reduce_mean.assert_called_with("sscewl")
            global_step.assert_called_with
            reduce_sum.assert_called_with('to_int32')
            argmax.assert_called_with("model", axis=1)
            get_train_ops.assert_called_with('global_step', mc.learning_rate, clip_mode=None, l2_reg=mc.l2_reg, num_train_batches=310, optim_algo=None)
            to_int32.assert_called_with('equal')
            equal.assert_called_with('to_int32', 2)

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.print')
    @patch('src.cifar10.macro_child.MacroChild.Model', return_value=mock.MagicMock(return_value='model'))
    @patch('src.cifar10.macro_child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.macro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.macro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.macro_child.fw.reduce_sum', return_value="reduce_sum")
    def test_build_valid(self, reduce_sum, equal, to_int32, argmax, model, print):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.x_valid = True
        mc.y_valid = None
        predictions, accuracy = mc._build_valid(mc.y_valid)
        logits = MacroChild.Model(mc, False, True)(mc.x_valid)
        self.assertEqual(('to_int32', 'reduce_sum'), (predictions(logits), accuracy(logits)))
        print.assert_any_call("-" * 80)
        print.assert_any_call("Build valid graph")
        model.assert_called_with(mc, False, True)
        model().assert_called_with(True)
        argmax.assert_called_with('model', axis=1)
        to_int32.assert_any_call('argmax')
        equal.assert_called_with('to_int32', None)
        to_int32.assert_any_call('equal')
        reduce_sum.assert_called_with('to_int32')

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.print')
    @patch('src.cifar10.macro_child.MacroChild.Model', return_value=mock.MagicMock(return_value='model'))
    @patch('src.cifar10.macro_child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.macro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.macro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.macro_child.fw.reduce_sum', return_value="reduce_sum")
    def test_build_test(self, reduce_sum, equal, to_int32, argmax, model, print):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.x_test = True
        mc.y_test = False
        predictions, accuracy = mc._build_test(mc.y_test)
        logits = MacroChild.Model(mc, False, True)(mc.x_test)
        self.assertEqual(('to_int32', 'reduce_sum'), (predictions(logits), accuracy(logits)))
        print.assert_any_call('-' * 80)
        print.assert_any_call("Build test graph")
        model.assert_called_with(mc, False, True)
        model().assert_called_with(True)
        argmax.assert_called_with('model', axis=1)
        to_int32.assert_any_call('argmax')
        equal.assert_called_with('to_int32', False)
        to_int32.assert_any_call('equal')
        reduce_sum.assert_called_with('to_int32')

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.MacroChild.Model', return_value=mock.MagicMock(return_value='model'))
    @patch('src.cifar10.macro_child.fw.map_fn', return_value="map_fn")
    @patch('src.cifar10.macro_child.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.macro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.macro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.macro_child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.macro_child.fw.shuffle_batch', return_value=(tf.constant(np.ndarray((2, 3, 32, 32))), "y_valid_shuffle"))
    @patch('src.cifar10.macro_child.print')
    def test_build_valid_rl(self, print, shuffle_batch, argmax, to_int32, equal, reduce_sum, map_fn, model):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.data_format = "NCHW"
        mc.images = { 'valid_original': np.ndarray((1, 3, 32, 32)) }
        mc.labels = { 'valid_original': np.ndarray((1)) }
        mc.batch_size = 32
        mc.seed = None
        shuffle = MacroChild.ValidationRLShuffle(mc, True)
        vrl = MacroChild.ValidationRL()
        x_valid_shuffle, y_valid_shuffle = shuffle(mc.images['valid_original'], mc.labels['valid_original'])
        logits = MacroChild.Model(mc, True, True)(x_valid_shuffle)
        vrl(logits, y_valid_shuffle)
        shuffle_batch.assert_called_with(
            [mc.images['valid_original'], mc.labels['valid_original']],
            mc.batch_size,
            mc.seed)
        model.assert_called_with(mc, True, True)
        model().assert_called_with('map_fn')
        argmax.assert_called_with("model", axis=1)
        to_int32.assert_any_call("argmax")
        equal.assert_called_with("to_int32", "y_valid_shuffle")
        to_int32.assert_any_call("equal")
        reduce_sum.assert_called_with("to_int32")
        map_fn.assert_called()

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    def test_connect_controller_no_fixed_arc(self):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        with patch.object(mc, '_build_train', return_value=('loss', 'train_loss', 'acc', 'gs', 'op', 'lr', 'gn', 'o')) as build_train:
            with patch.object(mc, '_build_valid', return_value=('predictions', 'accuracy')) as build_valid:
                with patch.object(mc, '_build_test', return_value=('predictions', 'accuracy')) as build_test:
                    controller_mock = mock.MagicMock()
                    mc.connect_controller(controller_mock)
                    build_train.assert_called_with(mc.y_train)
                    build_valid.assert_called_with(mc.y_valid)
                    build_test .assert_called_with(mc.y_test)

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    def test_connect_controller_fixed_arc(self):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.fixed_arc = ""
        with patch.object(mc, '_build_train', return_value=('loss', 'train_loss', 'acc', 'gs', 'op', 'lr', 'gn', 'o')) as build_train:
            with patch.object(mc, '_build_valid', return_value=('predictions', 'accuracy')) as build_valid:
                with patch.object(mc, '_build_test', return_value=('predictions', 'accuracy')) as build_test:
                    controller_mock = mock.MagicMock()
                    mc.connect_controller(controller_mock)
                    build_train.assert_called_with(mc.y_train)
                    build_valid.assert_called_with(mc.y_valid)
                    build_test .assert_called_with(mc.y_test)
