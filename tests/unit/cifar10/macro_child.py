import unittest
import unittest.mock as mock
from unittest.mock import patch

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import src.framework as fw

from src.cifar10.child import DataFormat
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
    self.lr_cosine = False
    self.lr_max = None
    self.lr_min = None
    self.lr_T_0 = None
    self.lr_T_mul = None

def mock_init_nhwc(self, images, labels, **kwargs):
    self.whole_channels = False
    self.data_format = DataFormat.new("NHWC")
    self.cutout_size = None
    self.num_layers = 2
    self.use_aux_heads = False
    self.num_train_batches = 1
    self.fixed_arc = None
    self.out_filters = 24
    self.lr_cosine = False
    self.lr_max = None
    self.lr_min = None
    self.lr_T_0 = None
    self.lr_T_mul = None

def mock_init_invalid(self, images, labels, **kwargs):
    self.whole_channels = False
    self.data_format = DataFormat.new("INVALID")


class TestMacroChild(unittest.TestCase):
    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    def test_init(self):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        self.assertEqual(MacroChild, type(mc))

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    def test_get_hw(self):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        self.assertEqual(mc._get_HW(fw.constant(np.ndarray((1, 2, 3)))), 3)

    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    def test_get_strides_nhwc(self):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        self.assertEqual([1, 2, 2, 1], mc._get_strides(2))

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    def test_get_strides_nchw(self):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        self.assertEqual([1, 1, 2, 2], mc._get_strides(2))

    @patch('src.cifar10.child.Child.__init__', new=mock_init_invalid)
    def test_get_strides_exception(self):
        with tf.Graph().as_default():
            self.assertRaises(KeyError, MacroChild, {}, {})

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    def test_factorized_reduction_failure(self):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        self.assertRaises(AssertionError, mc._factorized_reduction, None, 3, None, None)

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.create_weight')
    @patch('src.cifar10.macro_child.fw.conv2d', return_value=None)
    @patch('src.cifar10.macro_child.batch_norm')
    def test_factorized_reduction_stride1(self, batch_norm, conv2d, create_weight):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        self.assertRaises(AssertionError, mc._factorized_reduction, None, 3, None, None)
        with patch.object(mc.data_format, 'get_C', return_value=None) as get_C:
            mc._factorized_reduction(None, 2, 1, True)
            get_C.assert_called_with(None)
            create_weight.assert_called_with("w", [1, 1, None, 2])
            batch_norm.assert_called()

    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    @patch('src.cifar10.macro_child.fw.create_weight', return_value="w")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.fw.avg_pool', return_value="path1")
    @patch('src.cifar10.macro_child.fw.pad', return_value=np.ndarray((5, 5, 5, 5)))
    @patch('src.cifar10.macro_child.fw.concat', return_value="final_path")
    @patch('src.cifar10.macro_child.batch_norm', return_value="final_path")
    def test_factorized_reduction_nhwc(self, batch_norm, concat, pad, avg_pool, conv2d, create_weight):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        with patch.object(mc, '_get_strides', return_value="stride spec") as get_strides:
            with patch.object(mc.data_format, 'get_C', return_value="inp_c") as get_C:
                self.assertEqual("final_path", mc._factorized_reduction(None, 2, 2, True))
                get_strides.assert_called_with(2)
                avg_pool.assert_called()
                get_C.assert_any_call("path1")
                create_weight.assert_called_with("w", [1, 1, "inp_c", 1])
                conv2d.assert_called()
                pad.assert_called_with(None, [[0, 0], [0, 1], [0, 1], [0, 0]])
                avg_pool.assert_called()
                get_C.assert_any_call("path1")
                create_weight.assert_any_call("w", [1, 1, "inp_c", 1])
                conv2d.assert_called()
                concat.assert_called_with(values=["conv2d", "conv2d"], axis=3)
                batch_norm.assert_called()

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.create_weight', return_value="w")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.fw.avg_pool', return_value="path1")
    @patch('src.cifar10.macro_child.fw.pad', return_value=np.ndarray((5, 5, 5, 5)))
    @patch('src.cifar10.macro_child.fw.concat', return_value="final_path")
    @patch('src.cifar10.macro_child.batch_norm', return_value="final_path")
    def test_factorized_reduction_nchw(self, batch_norm, concat, pad, avg_pool, conv2d, create_weight):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        with patch.object(mc, '_get_strides', return_value="stride spec") as get_strides:
            with patch.object(mc.data_format, 'get_C', return_value="inp_c") as get_C:
                self.assertEqual("final_path", mc._factorized_reduction(None, 2, 2, True))
                get_strides.assert_called_with(2)
                avg_pool.assert_called()
                get_C.assert_any_call("path1")
                create_weight.assert_called_with("w", [1, 1, "inp_c", 1])
                conv2d.assert_called()
                pad.assert_called_with(None, [[0, 0], [0, 0], [0, 1], [0, 1]])
                avg_pool.assert_called()
                get_C.assert_any_call("path1")
                create_weight.assert_any_call("w", [1, 1, "inp_c", 1])
                conv2d.assert_called()
                concat.assert_called_with(values=["conv2d", "conv2d"], axis=1)
                batch_norm.assert_called()

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    def test_get_c_nchw(self):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        self.assertEqual(3, mc.data_format.get_C(tf.constant(np.ndarray((45000, 3, 32, 32)))))

    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    def test_get_c_nhwc(self):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        self.assertEqual(3, mc.data_format.get_C(tf.constant(np.ndarray((45000, 32, 32, 3)))))

    @patch('src.cifar10.child.Child.__init__', new=mock_init_invalid)
    def test_get_c_raises(self):
        with tf.Graph().as_default():
            self.assertRaises(KeyError, MacroChild, {}, {})

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.print')
    @patch('src.cifar10.macro_child.fw.create_weight', return_value="w")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.batch_norm', return_value="final_path")
    @patch('src.cifar10.macro_child.fw.dropout')
    def test_model_nhwc_KNOWN_TO_FAIL(self, dropout, batch_norm, conv2d, create_weight, _print):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.name = "generic_model"
        mc.keep_prob = 0.9
        with patch.object(mc.data_format, 'global_avg_pool') as global_avg_pool:
            with patch.object(mc, '_enas_layer') as enas_layer:
                mc._model({}, True)
                create_weight.assert_called_with("w", [3, 3, 3, 4])
                conv2d.assert_called()
                enas_layer.assert_called_with(0, 0, 0, 0, True)
                global_avg_pool.assert_called_with(None, data_format="NHWC")
                dropout.assert_called_with(None, 0.9)

    @patch('src.cifar10.macro_child.print')
    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.create_weight', return_value="w")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.batch_norm', return_value="final_path")
    @patch('src.cifar10.macro_child.fw.dropout', return_value=tf.constant(np.ndarray((4, 3, 32, 32))))
    @patch('src.cifar10.macro_child.fw.matmul')
    def test_model_nchw_no_whole_channels(self, matmul, dropout, batch_norm, conv2d, create_weight, _print):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.name = "generic_model"
        mc.keep_prob = 0.9
        mc.pool_layers = [0]
        with patch.object(mc.data_format, 'global_avg_pool', return_value="global_avg_pool") as global_avg_pool:
            with patch.object(mc, '_enas_layer', return_value="enas_layer") as enas_layer:
                with patch.object(mc, '_factorized_reduction', return_value="factorized_reduction") as factorized_reduction:
                    mc._model({}, True)
                    create_weight.assert_any_call("w", [3, 3, 3, 24])
                    conv2d.assert_called()
                    batch_norm.assert_called()
                    enas_layer.assert_called_with(1, ['factorized_reduction', 'factorized_reduction', 'enas_layer'], 18, 24, True)
                    factorized_reduction.assert_called_with('enas_layer', 24, 2, True)
                    global_avg_pool.assert_called()
                    dropout.assert_called_with('global_avg_pool', 0.9)
                    create_weight.assert_called_with("w", [3, 10])
                    matmul.assert_called()

    @patch('src.cifar10.macro_child.print')
    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.create_weight', return_value="w")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.batch_norm', return_value="final_path")
    @patch('src.cifar10.macro_child.fw.dropout', return_value=tf.constant(np.ndarray((4, 3, 32, 32))))
    @patch('src.cifar10.macro_child.fw.matmul')
    def test_model_nchw_whole_channels(self, matmul, dropout, batch_norm, conv2d, create_weight, _print):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.name = "generic_model"
        mc.keep_prob = 0.9
        mc.pool_layers = [0]
        mc.whole_channels = True
        mc.fixed_arc = ""
        with patch.object(mc.data_format, 'global_avg_pool', return_value='global_avg_pool') as global_avg_pool:
            with patch.object(mc, '_fixed_layer', return_value="enas_layer") as fixed_layer:
                with patch.object(mc, '_factorized_reduction', return_value="factorized_reduction") as factorized_reduction:
                    mc._model({}, True)
                    create_weight.assert_any_call("w", [3, 3, 3, 24])
                    conv2d.assert_called()
                    batch_norm.assert_called()
                    fixed_layer.assert_called_with(1, ['factorized_reduction', 'factorized_reduction', 'enas_layer'], 1, 48, True)
                    factorized_reduction.assert_called_with('enas_layer', 48, 2, True)
                    global_avg_pool.assert_called()
                    dropout.assert_called_with('global_avg_pool', 0.9)
                    create_weight.assert_called_with("w", [3, 10])
                    matmul.assert_called()

    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    @patch('src.cifar10.macro_child.fw.create_weight', return_value="w")
    @patch('src.cifar10.macro_child.fw.case', return_value=tf.constant(np.ndarray((4, 32, 32, 24)), tf.float32))
    @patch('src.cifar10.macro_child.fw.add_n', return_value="add_n")
    @patch('src.cifar10.macro_child.batch_norm', return_value="batch_norm")
    def test_enas_layer_whole_channels_nhwc(self, batch_norm, add_n, case, create_weight):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.whole_channels = True
        mc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0".split(" ") if x])
        with patch.object(mc, '_conv_branch', return_value="conv_branch") as conv_branch:
            with patch.object(mc, '_pool_branch', return_value="pool_branch") as pool_branch:
                input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
                mc._enas_layer(1, [input_tensor], 0, 24, True)
                conv_branch.assert_called_with(input_tensor, 5, True, 24, 24, start_idx=0, separable=True)
                case.assert_called()
                batch_norm.assert_called()
                create_weight.assert_not_called()

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.create_weight', return_value="w")
    @patch('src.cifar10.macro_child.fw.case', return_value=tf.constant(np.ndarray((4, 24, 32, 32)), tf.float32))
    @patch('src.cifar10.macro_child.fw.add_n', return_value="add_n")
    @patch('src.cifar10.macro_child.batch_norm', return_value="batch_norm")
    def test_enas_layer_whole_channels_nchw(self, batch_norm, add_n, case, create_weight):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.whole_channels = True
        mc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0".split(" ") if x])
        with patch.object(mc, '_conv_branch', return_value="conv_branch") as conv_branch:
            with patch.object(mc, '_pool_branch', return_value="pool_branch") as pool_branch:
                input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
                mc._enas_layer(1, [input_tensor], 0, 24, True)
                conv_branch.assert_called_with(input_tensor, 5, True, 24, 24, start_idx=0, separable=True)
                case.assert_called()
                batch_norm.assert_called()
                create_weight.assert_not_called()

    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    @patch('src.cifar10.macro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.macro_child.fw.boolean_mask', return_value="boolean_mask")
    @patch('src.cifar10.macro_child.fw.create_weight', return_value="w")
    @patch('src.cifar10.macro_child.fw.concat', return_value="concat")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.macro_child.fw.add_n', return_value="add_n")
    def test_enas_layer_not_whole_channels_nhwc(self, add_n, batch_norm, relu, conv_2d, concat, create_weight, boolean_mask, reshape):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.whole_channels = False
        mc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0 2 0 0 1 2 0 0 0 0".split(" ") if x])
        with patch.object(mc, '_conv_branch', return_value="conv_branch") as conv_branch:
            with patch.object(mc, '_pool_branch', return_value="pool_branch") as pool_branch:
                input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
                mc._enas_layer(1, [input_tensor], 0, 24, True)
                conv_branch.assert_called_with(input_tensor, 5, True, 0, 24, start_idx=2, separable=True)
                concat.assert_called()
                conv_2d.assert_called()
                relu.assert_called_with("batch_norm")
                batch_norm.assert_called()
                create_weight.assert_called_with('w', [144, 24])
                boolean_mask.assert_called()
                reshape.assert_called_with('boolean_mask', [1, 1, -1, 24])

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.concat', return_value="concat")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.macro_child.fw.add_n', return_value="add_n")
    @patch('src.cifar10.macro_child.fw.create_weight', return_value="fw.create_weight")
    @patch('src.cifar10.macro_child.fw.logical_and', return_value="logical_and")
    @patch('src.cifar10.macro_child.fw.logical_or', return_value="logical_or")
    @patch('src.cifar10.macro_child.fw.boolean_mask', return_value="boolean_mask")
    @patch('src.cifar10.macro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.macro_child.fw.shape', return_value=["shape"])
    def test_enas_layer_not_whole_channels_nchw(self, shape, reshape, boolean_mask, logical_or, logical_and, create_weight, add_n, batch_norm, relu, conv_2d, concat):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.whole_channels = False
        mc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0 2 0 0 1 2 0 0 0 0".split(" ") if x])
        with patch.object(mc, '_conv_branch', return_value="conv_branch") as conv_branch:
            with patch.object(mc, '_pool_branch', return_value="pool_branch") as pool_branch:
                input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
                mc._enas_layer(1, [input_tensor], 0, 24, True)
                conv_branch.assert_called_with(input_tensor, 5, True, 0, 24, start_idx=2, separable=True)
                concat.assert_called()
                conv_2d.assert_called()
                relu.assert_called_with("batch_norm")
                batch_norm.assert_called()
                create_weight.assert_called_with("w", [144, 24])
                logical_and.assert_called()
                logical_or.assert_called_with("logical_or", "logical_and")
                boolean_mask.assert_called_with("fw.create_weight", "logical_or")
                shape.assert_called()
                reshape.assert_called_with("concat", ['shape', -1, 32, 3])
                create_weight.assert_called_with('w', [144, 24])

    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    @patch('src.cifar10.macro_child.fw.create_weight', return_value="create_weight")
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.batch_norm', return_value="batch_norm")
    def test_fixed_layer_whole_channels_nhwc(self, batch_norm, conv2d, relu, create_weight):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.whole_channels = True
        mc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0".split(" ") if x])
        input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
        mc._fixed_layer(0, [input_tensor], 0, 24, True)
        relu.assert_called_with("batch_norm")
        conv2d.assert_called()
        batch_norm.assert_called()
        create_weight.assert_called_with('w', [3, 3, 24, 24])

    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.macro_child.fw.create_weight', return_value="fw.create_weight")
    @patch('src.cifar10.macro_child.fw.concat', return_value="concat")
    def test_fixed_layer_whole_channels_nhwc_second_layer(self, concat, create_weight, batch_norm, conv2d, relu):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.whole_channels = True
        mc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0".split(" ") if x])
        input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
        mc._fixed_layer(1, [input_tensor], 0, 24, True)
        create_weight.assert_called_with("w", [1, 1, 96, 24])
        relu.assert_called_with("concat")
        conv2d.assert_called()
        batch_norm.assert_called()

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.macro_child.fw.create_weight', return_value="fw.create_weight")
    def test_fixed_layer_whole_channels_nchw(self, create_weight, batch_norm, conv2d, relu):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.whole_channels = True
        mc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0".split(" ") if x])
        input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
        mc._fixed_layer(0, [input_tensor], 0, 24, True)
        relu.assert_called_with("batch_norm")
        conv2d.assert_called()
        batch_norm.assert_called()
        create_weight.assert_called_with('w', [3, 3, 24, 24])

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    def test_fixed_layer_whole_channels_nchw_raises(self):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.whole_channels = True
        mc.sample_arc = np.array([int(x) for x in "6 3 0 0 1 0".split(" ") if x])
        input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
        self.assertRaises(ValueError, mc._fixed_layer, 0, [input_tensor], 0, 24, True)

    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    @patch('src.cifar10.macro_child.fw.create_weight')
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.macro_child.fw.concat', return_value="concat")
    def test_fixed_layer_not_whole_channels_nhwc(self, concat, batch_norm, conv2d, relu, create_weight):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.whole_channels = False
        mc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0 2 0 0 1 2 0 0 0 0".split(" ") if x])
        input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
        with patch.object(mc, '_conv_branch', return_value="conv_branch") as conv_branch:
            with patch.object(mc, '_pool_branch', return_value="pool_branch") as pool_branch:
                mc._fixed_layer(0, [input_tensor], 0, 24, True)
                conv_branch.assert_called_with(input_tensor, 5, True, 0, separable=True)
                relu.assert_called_with("concat")
                conv2d.assert_called()
                batch_norm.assert_called()
                pool_branch.assert_called_with(input_tensor, True, 0, "max")
                concat.assert_called_with(['conv_branch', 'conv_branch', 'conv_branch', 'conv_branch', 'pool_branch', 'pool_branch'], axis=3)
                create_weight.assert_called_with('w', [1, 1, 4, 24])

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.create_weight')
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.macro_child.fw.concat', return_value="concat")
    def test_fixed_layer_not_whole_channels_nchw_second_layer(self, concat, batch_norm, conv2d, relu, create_weight):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.whole_channels = False
        mc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0 2 0 0 1 2 0 0 0 0".split(" ") if x])
        input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
        with patch.object(mc, '_conv_branch', return_value="conv_branch") as conv_branch:
            with patch.object(mc, '_pool_branch', return_value="pool_branch") as pool_branch:
                mc._fixed_layer(1, [input_tensor], 0, 24, True)
                conv_branch.assert_called_with(input_tensor, 5, True, 0, separable=True)
                relu.assert_called_with("concat")
                conv2d.assert_called()
                batch_norm.assert_called()
                pool_branch.assert_called_with(input_tensor, True, 0, "max")
                concat.assert_called_with(['batch_norm'], axis=1)
                create_weight.assert_called_with('w', [1, 1, 24, 24])

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    def test_conv_branch_failure(self):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        self.assertRaises(AssertionError, mc._conv_branch, None, 24, True, 0, 24)

    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.fw.create_weight', return_value="fw.create_weight")
    @patch('src.cifar10.macro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    def test_conv_branch_nhwc(self, relu, batch_norm, create_weight, conv2d):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.fixed_arc = "0 3 0 0 1 0"
        input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
        mc._conv_branch(input_tensor, 24, True, 0, 24)
        conv2d.assert_called()
        batch_norm.assert_called()
        relu.assert_called_with("batch_norm")
        create_weight.assert_called()

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.fw.create_weight', return_value="fw.create_weight")
    @patch('src.cifar10.macro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    def test_conv_branch_nchw(self, relu, batch_norm, create_weight, conv2d):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.fixed_arc = "0 3 0 0 1 0"
        input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
        mc._conv_branch(input_tensor, 24, True, 0, 24)
        create_weight.assert_any_call("w", [1, 1, 3, 24])
        conv2d.assert_called()
        batch_norm.assert_called()
        relu.assert_called_with("batch_norm")

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.fw.create_weight', return_value=tf.Variable(initial_value=np.zeros((24, 24, 24, 1))))
    @patch('src.cifar10.macro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.transpose', return_value="transpose")
    @patch('src.cifar10.macro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.macro_child.fw.separable_conv2d', return_value="sep_conv2d")
    @patch('src.cifar10.macro_child.batch_norm_with_mask', return_value="bnwm")
    def test_conv_branch_nchw_separable(self, bnwm, sep_conv2d, reshape, transpose, relu, batch_norm, create_weight, conv2d):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.fixed_arc = "0 3 0 0 1 0"
        mc.filter_size = 24
        input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
        mc._conv_branch(input_tensor, 24, True, 0, 24, ch_mul=1, start_idx=None, separable=True)
        create_weight.assert_any_call("w", [1, 1, 3, 24])
        conv2d.assert_called()
        batch_norm.assert_called()
        relu.assert_called_with("batch_norm")

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.fw.create_weight', return_value=tf.Variable(initial_value=np.zeros((24, 24, 24, 1))))
    @patch('src.cifar10.macro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.transpose', return_value=tf.Variable(initial_value=np.zeros((24, 24, 24, 1))))
    @patch('src.cifar10.macro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.macro_child.fw.separable_conv2d', return_value="sep_conv2d")
    @patch('src.cifar10.macro_child.batch_norm_with_mask', return_value="bnwm")
    def test_conv_branch_nchw_second_index(self, bnwm, sep_conv2d, reshape, transpose, relu, batch_norm, create_weight, conv2d):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.fixed_arc = "0 3 0 0 1 0"
        input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
        mc._conv_branch(input_tensor, 24, True, 0, 24, ch_mul=1, start_idx=1)
        create_weight.assert_any_call("w", [1, 1, 3, 24])
        conv2d.assert_called()
        batch_norm.assert_called()
        relu.assert_called_with("bnwm")

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.fw.create_weight', return_value=tf.Variable(initial_value=np.zeros((24, 24, 24, 1))))
    @patch('src.cifar10.macro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.transpose', return_value="transpose")
    @patch('src.cifar10.macro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.macro_child.fw.separable_conv2d', return_value="sep_conv2d")
    @patch('src.cifar10.macro_child.batch_norm_with_mask', return_value="bnwm")
    def test_conv_branch_nchw_second_index_separable(self, bnwm, sep_conv2d, reshape, transpose, relu, batch_norm, create_weight, conv2d):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.fixed_arc = "0 3 0 0 1 0"
        input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
        mc._conv_branch(input_tensor, 24, True, 0, 24, ch_mul=1, start_idx=1, separable=True)
        create_weight.assert_any_call("w", [1, 1, 3, 24])
        conv2d.assert_called()
        batch_norm.assert_called()
        relu.assert_called_with("bnwm")

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    def test_pool_branch_failure(self):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        self.assertRaises(AssertionError, mc._pool_branch, None, None, None, None)

    @patch('src.cifar10.child.Child.__init__', new=mock_init_nhwc)
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.avg_pool2d')
    @patch('src.cifar10.macro_child.fw.create_weight', return_value=tf.Variable(initial_value=np.zeros((24, 24, 24, 1))))
    def test_pool_branch_nhwc_avg(self, create_weight, avg_pool2d, relu, batch_norm, conv2d):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.fixed_arc = "0 3 0 0 1 0"
        input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
        mc._pool_branch(input_tensor, True, 0, "avg", start_idx=True)
        conv2d.assert_called()
        batch_norm.assert_called()
        relu.assert_called_with("batch_norm")
        avg_pool2d.assert_called_with("relu", [3, 3], [1, 1], "SAME", data_format="channels_last")
        create_weight.assert_called_with('w', [1, 1, 3, 24])

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.macro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.macro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.macro_child.fw.max_pool2d')
    @patch('src.cifar10.macro_child.fw.create_weight', return_value=tf.Variable(initial_value=np.zeros((24, 24, 24, 1))))
    def test_pool_branch_nchw_max(self, create_weight, max_pool2d, relu, batch_norm, conv2d):
        fw.FLAGS.controller_search_whole_channels = True
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.fixed_arc = "0 3 0 0 1 0"
        input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
        mc._pool_branch(input_tensor, True, 0, "max", start_idx=True)
        conv2d.assert_called()
        batch_norm.assert_called()
        relu.assert_called_with("batch_norm")
        max_pool2d.assert_called_with('relu', [3, 3], [1, 1], 'SAME', data_format='channels_first')
        create_weight.assert_called_with('w', [1, 1, 3, 24])

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.macro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.macro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.macro_child.print')
    @patch('src.cifar10.macro_child.fw.sparse_softmax_cross_entropy_with_logits', return_value="sscewl")
    @patch('src.cifar10.macro_child.fw.reduce_mean', return_value="reduce_mean")
    @patch('src.cifar10.macro_child.fw.argmax', return_value=10)
    @patch('src.cifar10.macro_child.get_train_ops', return_value=(1, 2, 3, 4))
    def test_build_train(self, get_train_ops, argmax, reduce_mean, sscewl, print, to_int32, equal, reduce_sum):
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
            with patch.object(mc, '_model', return_value="model") as model:
                mc._build_train()
                print.assert_any_call("-" * 80)
                print.assert_any_call("Build train graph")
                print.assert_called_with("Model has 0 params")
                model.assert_called_with(mc.x_train, is_training=True)
                sscewl.assert_called_with(logits="model", labels=mc.y_train)
                reduce_mean.assert_called_with("sscewl")
                argmax.assert_called_with("model", axis=1)
                get_train_ops.assert_called()
                to_int32.assert_called_with('equal')
                equal.assert_called_with('to_int32', 2)

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.print')
    @patch('src.cifar10.macro_child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.macro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.macro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.macro_child.fw.reduce_sum', return_value="reduce_sum")
    def test_build_valid(self, reduce_sum, equal, to_int32, argmax, print):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.x_valid = True
        mc.y_valid = None
        with patch.object(mc, '_model', return_value='model') as model:
            mc._build_valid()
            print.assert_any_call("-" * 80)
            print.assert_any_call("Build valid graph")
            model.assert_called_with(True, False, reuse=True)
            argmax.assert_called_with('model', axis=1)
            to_int32.assert_any_call('argmax')
            equal.assert_called_with('to_int32', None)
            to_int32.assert_any_call('equal')
            reduce_sum.assert_called_with('to_int32')

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.print')
    @patch('src.cifar10.macro_child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.macro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.macro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.macro_child.fw.reduce_sum', return_value="reduce_sum")
    def test_build_test(self, reduce_sum, equal, to_int32, argmax, print):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.x_test = True
        mc.y_test = False
        with patch.object(mc, '_model', return_value='model') as model:
            mc._build_test()
            print.assert_any_call('-' * 80)
            print.assert_any_call("Build test graph")
            model.assert_called_with(True, False, reuse=True)
            argmax.assert_called_with('model', axis=1)
            to_int32.assert_any_call('argmax')
            equal.assert_called_with('to_int32', False)
            to_int32.assert_any_call('equal')
            reduce_sum.assert_called_with('to_int32')

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    @patch('src.cifar10.macro_child.fw.map_fn', return_value="map_fn")
    @patch('src.cifar10.macro_child.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.macro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.macro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.macro_child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.macro_child.fw.shuffle_batch', return_value=(tf.constant(np.ndarray((2, 3, 32, 32))), "y_valid_shuffle"))
    @patch('src.cifar10.macro_child.print')
    def test_build_valid_rl(self, print, shuffle_batch, argmax, to_int32, equal, reduce_sum, map_fn):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.data_format = "NCHW"
        mc.images = { 'valid_original': np.ndarray((1, 3, 32, 32)) }
        mc.labels = { 'valid_original': np.ndarray((1)) }
        mc.batch_size = 32
        mc.seed = None
        with patch.object(mc, '_model', return_value='model') as model:
            mc.build_valid_rl(shuffle=True)
            print.assert_any_call('-' * 80)
            print.assert_any_call('Build valid graph on shuffled data')
            shuffle_batch.assert_called_with(
                [mc.images['valid_original'], mc.labels['valid_original']],
                mc.batch_size,
                mc.seed)
            model.assert_called()
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
        with patch.object(mc, '_build_train') as build_train:
            with patch.object(mc, '_build_valid') as build_valid:
                with patch.object(mc, '_build_test') as build_test:
                    controller_mock = mock.MagicMock()
                    mc.connect_controller(controller_mock)
                    build_train.assert_called_with()
                    build_valid.assert_called_with()
                    build_test.assert_called_with()

    @patch('src.cifar10.child.Child.__init__', new=mock_init)
    def test_connect_controller_fixed_arc(self):
        with tf.Graph().as_default():
            mc = MacroChild({}, {})
        mc.fixed_arc = ""
        with patch.object(mc, '_build_train') as build_train:
            with patch.object(mc, '_build_valid') as build_valid:
                with patch.object(mc, '_build_test') as build_test:
                    controller_mock = mock.MagicMock()
                    mc.connect_controller(controller_mock)
                    build_train.assert_called_with()
                    build_valid.assert_called_with()
                    build_test.assert_called_with()
