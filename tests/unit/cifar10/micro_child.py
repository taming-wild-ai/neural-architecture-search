import unittest
import unittest.mock as mock
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from src.cifar10.micro_child import MicroChild
from src.cifar10.child import DataFormat
from src.framework import WeightRegistry

def mock_init_nhwc(self, images, labels, **kwargs):
    self.data_format = DataFormat.new("NHWC")
    self.cutout_size = None
    self.num_layers = 2
    self.use_aux_heads = False
    self.num_train_batches = 1
    self.num_epochs = 310
    self.fixed_arc = None
    self.out_filters = 24
    self.learning_rate = mock.MagicMock()
    self.weights = WeightRegistry()

def mock_init_nchw(self, images, labels, **kwargs):
    self.data_format = DataFormat.new("NCHW")
    self.cutout_size = None
    self.num_layers = 2
    self.use_aux_heads = False
    self.num_train_batches = 1
    self.num_epochs = 310
    self.fixed_arc = None
    self.out_filters = 24
    self.learning_rate = mock.MagicMock()
    self.weights = WeightRegistry()

def mock_init_invalid_data_format(self, images, labels, **kwargs):
    self.data_format = DataFormat.new("INVALID")
    self.cutout_size = None
    self.num_layers = 2
    self.use_aux_heads = False
    self.num_train_batches = 1
    self.num_epochs = 310
    self.fixed_arc = None
    self.out_filters = 24
    self.lr_cosine = False
    self.lr_max = None
    self.lr_min = None
    self.lr_T_0 = None
    self.lr_T_mul = None
    self.weights = WeightRegistry()


class TestMicroChild(unittest.TestCase):
    def test_micro_child_nhwc(self):
        with tf.Graph().as_default():
            with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
                mc = MicroChild({}, {})
                self.assertEqual(MicroChild, type(mc))

    def test_micro_child_nchw(self):
        with tf.Graph().as_default():
            with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
                mc = MicroChild({}, {}, data_format="NCHW", use_aux_heads=True)
                self.assertEqual(MicroChild, type(mc))

    def test_micro_child_raises(self):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_invalid_data_format):
            self.assertRaises(KeyError, MicroChild, {}, {})

    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.concat', return_value="concat")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.avg_pool', return_value="avg_pool")
    @patch('src.cifar10.micro_child.fw.pad', return_value=tf.constant(np.ones((4, 32, 32, 3))))
    def test_factorized_reduction_nhwc_stride2(self, pad, avg_pool, conv2d, concat, batch_norm):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            with patch.object(mc.data_format, "get_C", return_value=3) as getc:
                with patch.object(mc.data_format, "get_strides", return_value="get_strides") as gets:
                    with patch.object(mc.weights, "create_weight", return_value="fw.create_weight") as create_weight:
                        retval = mc._factorized_reduction(None, 24, 2, True, mc.weights, False)
                        gets.assert_called_with(2)
                        pad.assert_called_with(None, [[0, 0], [0, 1], [0, 1], [0, 0]])
                        avg_pool.assert_called()
                        getc.assert_called_with('avg_pool')
                        create_weight.assert_called_with('w', [1, 1, 3, 12], initializer=None, trainable=True)
                        conv2d.assert_called_with('avg_pool', 'fw.create_weight', [1, 1, 1, 1], "VALID", data_format="NHWC")
                        concat.assert_called_with(values=['conv2d', 'conv2d'], axis=3)
                        batch_norm.assert_called()
                        self.assertEqual("batch_norm", retval)

    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.concat', return_value="concat")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.avg_pool', return_value="avg_pool")
    @patch('src.cifar10.micro_child.fw.pad', return_value=tf.constant(np.ones((4, 32, 32, 3))))
    def test_factorized_reduction_nchw_stride2(self, pad, avg_pool, conv2d, concat, batch_norm):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nchw):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            with patch.object(mc.data_format, "get_C", return_value=3) as getc:
                with patch.object(mc.data_format, "get_strides", return_value="get_strides") as gets:
                    with patch.object(mc.weights, "get", return_value="fw.create_weight") as create_weight:
                        retval = mc._factorized_reduction(None, 24, 2, True, mc.weights, False)
                        gets.assert_called_with(2)
                        pad.assert_called_with(None, [[0, 0], [0, 0], [0, 1], [0, 1]])
                        avg_pool.assert_called()
                        getc.assert_called_with('avg_pool')
                        create_weight.assert_called_with('w', [1, 1, 3, 12], None, False)
                        conv2d.assert_called_with('avg_pool', 'fw.create_weight', [1, 1, 1, 1], "VALID", data_format="NCHW")
                        concat.assert_called_with(values=['conv2d', 'conv2d'], axis=1)
                        batch_norm.assert_called()
                        self.assertEqual("batch_norm", retval)

    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    def test_factorized_reduction_nchw_stride1(self, conv2d, batch_norm):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nchw):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            with patch.object(mc.data_format, "get_C", return_value=3) as getc:
                with patch.object(mc.data_format, "get_strides", return_value="get_strides") as gets:
                    with patch.object(mc.weights, "get", return_value="fw.create_weight") as create_weight:
                        retval = mc._factorized_reduction(None, 24, 1, True, mc.weights, False)
                        create_weight.assert_called_with("w", [1, 1, 3, 24], None, False)
                        conv2d.assert_called_with(None, "fw.create_weight", [1, 1, 1, 1], 'SAME', data_format="NCHW")
                        batch_norm.assert_called()

    def test_get_c_nchw(self):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nchw):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            self.assertEqual(3, mc.data_format.get_C(tf.constant(np.ndarray((45000, 3, 32, 32)))))

    def test_get_c_nhwc(self):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            self.assertEqual(3, mc.data_format.get_C(tf.constant(np.ndarray((45000, 32, 32, 3)))))

    def test_get_hw(self):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            self.assertEqual(mc._get_HW(tf.constant(np.ndarray((1, 2, 3)))), 3)

    def test_get_strides_nhwc(self):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            self.assertEqual([1, 2, 2, 1], mc.data_format.get_strides(2))

    def test_get_strides_nchw(self):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nchw):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            self.assertEqual([1, 1, 2, 2], mc.data_format.get_strides(2))

    @patch('src.cifar10.macro_controller.fw.to_float', return_value=1.0)
    @patch('src.cifar10.macro_controller.fw.divide', return_value=1.0)
    @patch('src.cifar10.macro_controller.fw.shape', return_value=[5])
    def test_apply_drop_path(self, shape, divide, to_float):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
                mc._apply_drop_path(None, 0)
                shape.assert_called_with(None)
                divide.assert_called()
                to_float.assert_called_with(310)

    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    def test_maybe_calibrate_size_same(self, relu, conv2d, batch_norm):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            with patch.object(mc.weights, 'get', return_value="fw.create_weight") as create_weight:
                mc._maybe_calibrate_size([
                    tf.constant(np.ndarray((45000, 32, 32, 3))),
                    tf.constant(np.ndarray((45000, 32, 32, 3)))], 24, True, mc.weights, False)
                create_weight.assert_called_with('w', [1, 1, 3, 24], None, False)
                relu.assert_called()
                conv2d.assert_called_with('relu', 'fw.create_weight', [1, 1, 1, 1], 'SAME', data_format='NHWC')
                batch_norm.assert_called()

    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    def test_maybe_calibrate_size_different(self, relu, conv2d, batch_norm):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            with patch.object(mc, '_factorized_reduction', return_value="fr"):
                with patch.object(mc.weights, "get", return_value="fw.create_weight") as create_weight:
                    mc._maybe_calibrate_size([
                        tf.constant(np.ndarray((45000, 16, 32, 3))),
                        tf.constant(np.ndarray((45000, 32, 16, 3)))], 24, True, mc.weights, False)
                    create_weight.assert_called_with('w', [1, 1, 3, 24], None, False)
                    relu.assert_called()
                    conv2d.assert_called_with('relu', 'fw.create_weight', [1, 1, 1, 1], 'SAME', data_format="NHWC")
                    batch_norm.assert_called()

    @patch('src.cifar10.micro_child.fw.matmul', return_value="matmul")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.print')
    def test_model_nhwc(self, print, conv2d, batch_norm, relu, matmul):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            mc.name = "MicroChild"
            mc.reduce_arc = None
            mc.normal_arc = None
            mc.keep_prob = None
            with patch.object(mc, '_factorized_reduction', return_value="fr") as fr:
                with patch.object(mc, '_enas_layer', return_value="el") as el:
                    with patch.object(mc, 'data_format') as data_format:
                        with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                            mc._model({}, True)
                            create_weight.assert_called_with("w", [data_format.get_C(), 10], None, False)
                            conv2d.assert_called_with({}, "fw.create_weight", [1, 1, 1, 1], 'SAME', data_format=data_format.name)
                            batch_norm.assert_called_with("conv2d", True, data_format, mc.weights)
                            fr.assert_called_with('el', 96, 2, True, mc.weights, False)
                            el.assert_called_with(3, ['el', 'el'], None, 96, mc.weights, False)
                            data_format.global_avg_pool.assert_called_with('relu')
                            relu.assert_called_with('el')
                            data_format.get_C.assert_called_with()
                            matmul.assert_called_with(data_format.global_avg_pool(), 'fw.create_weight')

    @patch('src.cifar10.micro_child.fw.matmul', return_value="matmul")
    @patch('src.cifar10.micro_child.fw.dropout', return_value="dropout")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.print')
    def test_model_nchw(self, print, conv2d, batch_norm, relu, do, matmul):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nchw):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            with patch.object(mc.data_format, 'global_avg_pool', return_value="gap") as gap:
                with patch.object(mc, '_factorized_reduction', return_value="fr") as fr:
                    with patch.object(mc, '_enas_layer', return_value="el") as el:
                        with patch.object(mc.data_format, 'get_C', return_value="get_c") as get_c:
                            with patch.object(mc.weights, 'get', return_value="fw.create_weight") as create_weight:
                                mc.name = "MicroChild"
                                mc.reduce_arc = None
                                mc.normal_arc = None
                                mc.keep_prob = 0.9
                                mc._model({}, True)
                                create_weight.assert_any_call("w", [3, 3, 3, 72], None, False)
                                conv2d.assert_called_with({}, "fw.create_weight", [1, 1, 1, 1], 'SAME', data_format="NCHW")
                                batch_norm.assert_called()
                                fr.assert_called_with('el', 96, 2, True, mc.weights, False)
                                el.assert_called_with(3, ['el', 'el'], None, 96, mc.weights, False)
                                for num in range(4):
                                    print.assert_any_call(f"Layer  {num}: el")
                                relu.assert_called_with('el')
                                gap.assert_called_with("relu")
                                do.assert_called_with("gap", 0.9)
                                get_c.assert_called_with('dropout')
                                create_weight.assert_called_with("w", ["get_c", 10], None, False)
                                matmul.assert_called_with('dropout', "fw.create_weight")

    @patch('src.cifar10.micro_child.fw.avg_pool2d')
    @patch('src.cifar10.micro_child.fw.matmul', return_value="matmul")
    @patch('src.cifar10.micro_child.fw.dropout', return_value="dropout")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.print')
    def test_model_aux_heads(self, print, conv2d, batch_norm, relu, do, matmul, avg_pool2d):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nchw):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            with patch.object(mc.data_format, 'global_avg_pool', return_value=tf.constant(np.ndarray((3, 3, 32, 32)))) as gap:
                with patch.object(mc, '_factorized_reduction', return_value="fr") as fr:
                    with patch.object(mc, '_enas_layer', return_value="el") as el:
                        with patch.object(mc.data_format, 'get_C', return_value="get_c") as get_c:
                            with patch.object(mc, "_get_HW", return_value="get_hw") as get_hw:
                                with patch.object(mc.weights, "get", return_value="fw.create_weight") as create_weight:
                                    mc.name = "MicroChild"
                                    mc.reduce_arc = None
                                    mc.normal_arc = None
                                    mc.keep_prob = 0.9
                                    mc.use_aux_heads = True
                                    mc.aux_head_indices = [0]
                                    mc._model({}, True)
                                    create_weight.assert_any_call("w", [3, 3, 3, 72], None, False)
                                    conv2d.assert_called_with("relu", "fw.create_weight", [1, 1, 1, 1], 'SAME', data_format="NCHW")
                                    batch_norm.assert_called()
                                    fr.assert_called_with('el', 96, 2, True, mc.weights, False)
                                    el.assert_called_with(3, ['el', 'el'], None, 96, mc.weights, False)
                                    for num in range(4):
                                        print.assert_any_call(f"Layer  {num}: el")
                                    relu.assert_called_with('el')
                                    gap.assert_called_with("relu")
                                    do.assert_called()
                                    get_c.assert_called_with('dropout')
                                    create_weight.assert_called_with("w", ["get_c", 10], None, False)
                                    matmul.assert_called_with('dropout', "fw.create_weight")
                                    avg_pool2d.assert_called_with("relu", [5, 5], [3, 3], "VALID", data_format="channels_first")
                                    get_hw.assert_called_with("relu")

    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.separable_conv2d', return_value="s_conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    def test_fixed_conv(self, relu, s_conv2d, batch_norm):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            with patch.object(mc.data_format, 'get_C', return_value="get_c") as get_c:
                with patch.object(mc.weights, 'get', return_value="fw.create_weight") as create_weight:
                    mc._fixed_conv(None, 3, 24, 1, True, mc.weights, False)
                    get_c.assert_called_with('batch_norm')
                    # create_weight.assert_any_call("w_depth", [3, 3, "get_c", 1])
                    create_weight.assert_called_with("w_point", [1, 1, "get_c", 24], None, False)
                    relu.assert_called_with("batch_norm")
                    s_conv2d.assert_called_with("relu", depthwise_filter="fw.create_weight", pointwise_filter="fw.create_weight", strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
                    batch_norm.assert_called()

    def test_fixed_combine_small_hw(self):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nchw):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            with patch.object(mc, '_get_HW', return_value=32) as get_HW:
                mc._fixed_combine([1], [0], 24, True)
                get_HW.assert_called_with(1)

    @patch('src.cifar10.micro_child.fw.concat', return_value="concat")
    def test_fixed_combine_large_hw(self, concat):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            with patch.object(mc, '_factorized_reduction', return_value='fr') as fr:
                mc._fixed_combine([tf.constant(np.ndarray((1, 32, 32, 3))), tf.constant(np.ndarray((1, 64, 64, 3)))], [0, 0], 24, True)
                fr.assert_called()
                concat.assert_called()

    @patch('src.cifar10.micro_child.fw.max_pool2d')
    @patch('src.cifar10.micro_child.fw.avg_pool2d')
    @patch('src.cifar10.micro_child.fw.separable_conv2d', return_value="s_conv2d")
    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    def test_fixed_layer(self, relu, conv2d, batch_norm, s_conv2d, avg_pool2d, max_pool2d):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            with patch.object(mc, '_maybe_calibrate_size', return_value=[0, 1]) as mcs:
                with patch.object(mc.data_format, 'get_C', return_value="get_c") as get_c:
                    with patch.object(mc, '_apply_drop_path', return_value="adp") as adp:
                        with patch.object(mc, '_fixed_combine', return_value="fc") as fc:
                            with patch.object(mc.weights, 'get', return_value="fw.create_weight") as create_weight:
                                self.assertEqual("fc", mc._fixed_layer(0, [tf.constant(np.ndarray((1, 32, 32, 3)))] * 2, [0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 2, 0, 3, 0, 3, 0, 4, 0, 4], 24, 1, True, mc.weights, False))
                                mcs.assert_called()
                                get_c.assert_called_with(0)
                                create_weight.assert_called_with("w", [1, 1, 'get_c', 24], None, False)
                                relu.assert_called_with(0)
                                conv2d.assert_called_with("relu", "fw.create_weight", [1, 1, 1, 1], "SAME", data_format="NHWC")
                                batch_norm.assert_called()
                                s_conv2d.assert_called_with("relu", depthwise_filter="fw.create_weight", pointwise_filter="fw.create_weight", strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
                                adp.assert_called_with('batch_norm', 0)
                                fc.assert_called()

    @patch('src.cifar10.micro_child.fw.stack', return_value=tf.constant(np.ndarray((1, 4, 32, 32, 3))))
    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.micro_child.fw.max_pool2d', return_value="max_pool2d")
    @patch('src.cifar10.micro_child.fw.avg_pool2d', return_value="avg_pool2d")
    def test_enas_cell(self, avg_pool2d, max_pool2d, reshape, relu, conv2d, batch_norm, stack):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            with patch.object(mc.data_format, 'get_C', return_value="get_c") as get_c:
                with patch.object(mc, '_enas_conv', return_value="ec") as ec:
                    with patch.object(mc.weights, 'get', return_value="fw.create_weight") as create_weight:
                        mc._enas_cell(None, 0, 1, 0, 24, mc.weights, False)
                        avg_pool2d.assert_called_with(None, [3, 3], [1, 1], "SAME", data_format="channels_last")
                        get_c.assert_called_with(None)
                        create_weight.assert_called_with("w", [1, 'get_c' * 24], None, False)
                        reshape.assert_called_with('w', [1, 1, 'get_c', 24])
                        relu.assert_called_with(None)
                        conv2d.assert_called_with("relu", 'reshape', strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
                        batch_norm.assert_called()
                        ec.assert_called_with('batch_norm', 0, 1, 5, 24, mc.weights, False)
                        stack.assert_called_with(['ec', 'ec', 'batch_norm', 'batch_norm', 'batch_norm'], axis=0)

    @patch('src.cifar10.micro_child.fw.ones_init', return_value="ones")
    @patch('src.cifar10.micro_child.fw.zeros_init', return_value="zeros")
    @patch('src.cifar10.micro_child.fw.fused_batch_norm', return_value="fbn")
    @patch('src.cifar10.micro_child.fw.separable_conv2d', return_value="s_conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.fw.reshape', return_value="reshape")
    def test_enas_conv(self, reshape, relu, s_conv, fbn, zeros, ones):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            with patch.object(mc.data_format, 'get_C', return_value="get_c") as get_c:
                with patch.object(mc.weights, 'get', return_value=tf.constant(np.ndarray((3, 3)))) as create_weight:
                    mc._enas_conv(None, 0, 0, 3, 24, mc.weights, False)
                    get_c.assert_called_with('f')
                    create_weight.assert_called()
                    reshape.assert_called()
                    relu.assert_called_with('f')
                    s_conv.assert_called_with('relu', depthwise_filter='reshape', pointwise_filter='reshape', strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
                    fbn.assert_called()
                    zeros.assert_called_with()
                    ones.assert_called_with()

    @patch('src.cifar10.micro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.micro_child.fw.transpose', return_value="transpose")
    @patch('src.cifar10.micro_child.fw.gather', return_value="gather")
    @patch('src.cifar10.micro_child.fw.stack', return_value=tf.constant(np.ndarray((2, 3, 32, 32, 3))))
    def test_enas_layer_nhwc(self, stack, gather, transpose, reshape, relu, conv2d, batch_norm, to_int32):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            with patch.object(mc, '_maybe_calibrate_size', return_value=[tf.constant(np.ndarray((3, 32, 32, 3))), tf.constant(np.ndarray((3, 32, 32, 3)))]) as mcs:
                with patch.object(mc, '_enas_cell', return_value='ec') as ec:
                    with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                        mc._enas_layer(0, [tf.constant(np.ndarray((3, 32, 32, 3))), tf.constant(np.ndarray((3, 32, 32, 3)))], [0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 2, 0, 3, 0, 3, 0, 4, 0, 4], 3, mc.weights, False)
                        mcs.assert_called()
                        ec.assert_called()
                        stack.assert_called()
                        gather.assert_called()
                        transpose.assert_called_with('gather', [1, 2, 3, 0, 4])
                        reshape.assert_called()
                        relu.assert_called_with('reshape')
                        conv2d.assert_called_with('relu', 'reshape', strides=[1, 1, 1, 1], padding='SAME', data_format="NHWC")
                        batch_norm.assert_called()
                        create_weight.assert_called_with('w', [7, 9], None, False)
                        to_int32.assert_called()

    @patch('src.cifar10.micro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.micro_child.fw.transpose', return_value="transpose")
    @patch('src.cifar10.micro_child.fw.gather', return_value="gather")
    @patch('src.cifar10.micro_child.fw.stack', return_value=tf.constant(np.ndarray((2, 3, 32, 32, 3))))
    def test_enas_layer_nchw(self, stack, gather, transpose, reshape, relu, conv2d, batch_norm, to_int32):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nchw):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            with patch.object(mc, '_maybe_calibrate_size', return_value=[tf.constant(np.ndarray((3, 32, 32, 3))), tf.constant(np.ndarray((3, 32, 32, 3)))]) as mcs:
                with patch.object(mc, '_enas_cell', return_value='ec') as ec:
                    with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                        mc._enas_layer(0, [tf.constant(np.ndarray((3, 32, 32, 3))), tf.constant(np.ndarray((3, 32, 32, 3)))], [0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 2, 0, 3, 0, 3, 0, 4, 0, 4], 3, mc.weights, False)
                        mcs.assert_called()
                        ec.assert_called()
                        stack.assert_called()
                        gather.assert_called()
                        transpose.assert_called_with('gather', [1, 0, 2, 3, 4])
                        reshape.assert_called()
                        relu.assert_called_with('reshape')
                        conv2d.assert_called_with('relu', 'reshape', strides=[1, 1, 1, 1], padding='SAME', data_format="NCHW")
                        batch_norm.assert_called()
                        to_int32.assert_called()

    @patch('src.cifar10.micro_child.get_train_ops', return_value=(None, None, None, None))
    @patch('src.cifar10.micro_child.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.micro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.micro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.micro_child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.micro_child.fw.reduce_mean', return_value=1.0)
    @patch('src.cifar10.micro_child.fw.sparse_softmax_cross_entropy_with_logits', return_value="sscewl")
    @patch('src.cifar10.micro_child.print')
    def test_build_train_aux_heads(self, print, sscewl, reduce_mean, argmax, to_int32, equal, reduce_sum, get_train_ops):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            mc.name = "MicroChild"
            mc.use_aux_heads = True
            mc.x_train = {}
            mc.y_train = {}
            mc.aux_logits = None
            mc.clip_mode = None
            mc.grad_bound = None
            mc.l2_reg = None
            mc.lr_init = None
            mc.lr_dec_start = None
            mc.lr_dec_every = None
            mc.lr_dec_rate = None
            mc.optim_algo = None
            mc.sync_replicas = None
            mc.sync_replicas = None
            mc.num_aggregate = None
            mc.num_replicas = None
            with patch.object(mc, '_model', return_value="model") as model:
                mc._build_train()
                print.assert_any_call('-' * 80)
                print.assert_any_call("Build train graph")
                model.assert_called_with({}, is_training=True)
                sscewl.assert_called_with(logits=None, labels={})
                reduce_mean.assert_called_with("sscewl")
                argmax.assert_called_with("model", axis=1)
                to_int32.assert_called_with("equal")
                equal.assert_called_with("to_int32", {})
                reduce_sum.assert_called_with("to_int32")
                get_train_ops.assert_called()

    @patch('src.cifar10.micro_child.get_train_ops', return_value=(None, None, None, None))
    @patch('src.cifar10.micro_child.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.micro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.micro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.micro_child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.micro_child.fw.reduce_mean', return_value=1.0)
    @patch('src.cifar10.micro_child.fw.sparse_softmax_cross_entropy_with_logits', return_value="sscewl")
    @patch('src.cifar10.micro_child.print')
    def test_build_train_no_aux_heads(self, print, sscewl, reduce_mean, argmax, to_int32, equal, reduce_sum, get_train_ops):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            mc.name = "MicroChild"
            mc.use_aux_heads = False
            mc.x_train = {}
            mc.y_train = {}
            mc.clip_mode = None
            mc.grad_bound = None
            mc.l2_reg = None
            mc.lr_init = None
            mc.lr_dec_start = None
            mc.lr_dec_every = None
            mc.lr_dec_rate = None
            mc.optim_algo = None
            mc.sync_replicas = None
            mc.num_aggregate = None
            mc.num_replicas = None
            with patch.object(mc, '_model', return_value="model") as model:
                mc._build_train()
                print.assert_any_call('-' * 80)
                print.assert_any_call("Build train graph")
                model.assert_called_with({}, is_training=True)
                sscewl.assert_called_with(logits="model", labels={})
                reduce_mean.assert_called_with("sscewl")
                argmax.assert_called_with("model", axis=1)
                to_int32.assert_called_with("equal")
                equal.assert_called_with("to_int32", {})
                reduce_sum.assert_called_with("to_int32")
                get_train_ops.assert_called()

    @patch('src.cifar10.micro_child.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.micro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.micro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.micro_child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.micro_child.print')
    def test_build_valid(self, print, argmax, to_int32, equal, reduce_sum):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            mc.x_valid = {}
            mc.y_valid = {}
            with patch.object(mc, '_model', return_value="model") as model:
                mc._build_valid()
                print.assert_any_call('-' * 80)
                print.assert_any_call("Build valid graph")
                argmax.assert_called_with("model", axis=1)
                to_int32.assert_any_call("argmax")
                equal.assert_any_call("to_int32", {})
                to_int32.assert_any_call("equal")
                reduce_sum.assert_called_with("to_int32")

    @patch('src.cifar10.micro_child.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.micro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.micro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.micro_child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.micro_child.print')
    def test_build_test(self, print, argmax, to_int32, equal, reduce_sum):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            mc.x_test = {}
            mc.y_test = {}
            with patch.object(mc, '_model', return_value="model") as model:
                mc._build_test()
                print.assert_any_call('-' * 80)
                print.assert_any_call("Build test graph")
                argmax.assert_called_with("model", axis=1)
                to_int32.assert_any_call("argmax")
                equal.assert_any_call("to_int32", {})
                to_int32.assert_any_call("equal")
                reduce_sum.assert_called_with("to_int32")

    @patch('src.cifar10.child.Child.__init__')
    @patch('src.cifar10.micro_child.fw.map_fn', return_value="map_fn")
    @patch('src.cifar10.micro_child.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.micro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.micro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.micro_child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.micro_child.fw.shuffle_batch', return_value=(tf.constant(np.ndarray((2, 32, 32, 3))), "y_valid_shuffle"))
    @patch('src.cifar10.micro_child.print')
    def test_build_valid_rl(self, print, shuffle_batch, argmax, to_int32, equal, reduce_sum, map_fn, _Child):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            mc.images = { 'valid_original': np.ndarray((1, 32, 32, 3)) }
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
                    mc.seed,
                    25000)
                model.assert_called()
                argmax.assert_called_with("model", axis=1)
                to_int32.assert_any_call("argmax")
                equal.assert_called_with("to_int32", "y_valid_shuffle")
                to_int32.assert_any_call("equal")
                reduce_sum.assert_called_with("to_int32")
                map_fn.assert_called()

    def test_connect_controller(self):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            with patch.object(mc, '_build_train', return_value='model') as build_train:
                with patch.object(mc, '_build_valid', return_value='model') as build_valid:
                    with patch.object(mc, '_build_test', return_value='model') as build_test:
                        mock_controller = mock.MagicMock()
                        mock_controller.sample_arc = (None, None)
                        mc.connect_controller(mock_controller)

if "__main__" == "__name__":
    unittest.main()
