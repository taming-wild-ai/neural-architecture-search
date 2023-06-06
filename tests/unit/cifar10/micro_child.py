import sys
import unittest
import unittest.mock as mock
from unittest.mock import patch

import numpy as np

from src.cifar10.micro_child import MicroChild
from src.cifar10.child import Child, DataFormat
import src.framework as fw

import tensorflow as tf
from absl import flags
flags.FLAGS(['test'])

def mock_init_nhwc(self, images, labels, **kwargs):
    self.data_format = DataFormat.new("NHWC")
    self.cutout_size = None
    self.num_layers = 2
    self.use_aux_heads = False
    self.num_train_batches = 1
    self.num_epochs = 310
    self.fixed_arc = None
    self.out_filters = 24
    self.learning_rate = mock.MagicMock(name='learning_rate')
    self.weights = fw.WeightRegistry()
    self.dataset = None
    self.dataset_valid = None
    self.dataset_test = None
    self.global_step = 0

def mock_init_nchw(self, images, labels, **kwargs):
    self.data_format = DataFormat.new("NCHW")
    self.cutout_size = None
    self.num_layers = 2
    self.use_aux_heads = False
    self.num_train_batches = 1
    self.num_epochs = 310
    self.fixed_arc = None
    self.out_filters = 24
    self.learning_rate = mock.MagicMock(name='learning_rate')
    self.weights = fw.WeightRegistry()
    self.dataset = None
    self.dataset_valid = None
    self.dataset_test = None
    self.global_step = 0

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
    self.weights = fw.WeightRegistry()
    self.dataset_valid = None
    self.dataset_test = None


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

    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.micro_child.fw.concat', return_value="concat")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.avg_pool', return_value="avg_pool")
    @patch('src.cifar10.micro_child.fw.pad')
    def test_factorized_reduction_nhwc_stride2(self, pad, avg_pool, conv2d, concat, batch_norm):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
                with patch.object(mc.data_format, "get_strides", return_value="get_strides") as gets:
                    with patch.object(mc.weights, "create_weight", return_value="fw.create_weight") as create_weight:
                        fr = Child.FactorizedReduction(mc, 3, 24, 2, True, mc.weights, False)
                        retval = fr(None)
                        gets.assert_called_with(2)
                        pad.assert_called_with(None, [[0, 0], [0, 1], [0, 1], [0, 0]])
                        avg_pool.assert_called_with(pad().__getitem__(), [1, 1, 1, 1], 'get_strides', 'VALID', data_format='NHWC')
                        create_weight.assert_called_with('path2_conv/', 'w', [1, 1, 3, 12], initializer=None, trainable=True)
                        conv2d.assert_called_with('avg_pool', 'fw.create_weight', [1, 1, 1, 1], "VALID", data_format="NHWC")
                        concat.assert_called_with(values=['conv2d', 'conv2d'], axis=3)
                        batch_norm.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                        batch_norm().assert_called_with('concat')
                        self.assertEqual("batch_norm", retval)

    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.micro_child.fw.concat', return_value="concat")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.avg_pool', return_value="avg_pool")
    @patch('src.cifar10.micro_child.fw.pad')
    def test_factorized_reduction_nchw_stride2(self, pad, avg_pool, conv2d, concat, batch_norm):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nchw):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
                with patch.object(mc.data_format, "get_strides", return_value="get_strides") as gets:
                    with patch.object(mc.weights, "get", return_value="fw.create_weight") as create_weight:
                        fr = Child.FactorizedReduction(mc, 3, 24, 2, True, mc.weights, False)
                        retval = fr(None)
                        gets.assert_called_with(2)
                        pad.assert_called_with(None, [[0, 0], [0, 0], [0, 1], [0, 1]])
                        avg_pool.assert_called_with(pad().__getitem__(), [1, 1, 1, 1], 'get_strides', 'VALID', data_format='NCHW')
                        create_weight.assert_called_with(False, 'path2_conv/', 'w', [1, 1, 3, 12], None)
                        conv2d.assert_called_with('avg_pool', 'fw.create_weight', [1, 1, 1, 1], "VALID", data_format="NCHW")
                        concat.assert_called_with(values=['conv2d', 'conv2d'], axis=1)
                        batch_norm.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                        batch_norm().assert_called_with('concat')
                        self.assertEqual("batch_norm", retval)

    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    def test_factorized_reduction_nchw_stride1(self, conv2d, batch_norm):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nchw):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
                with patch.object(mc.weights, "get", return_value="fw.create_weight") as create_weight:
                    fr = Child.FactorizedReduction(mc, 3, 24, 1, True, mc.weights, False)
                    _ = fr(None)
                    create_weight.assert_called_with(False, 'path_conv/', "w", [1, 1, 3, 24], None)
                    conv2d.assert_called_with(None, "fw.create_weight", [1, 1, 1, 1], 'SAME', data_format="NCHW")
                    batch_norm.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                    batch_norm().assert_called_with('conv2d')

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

    @patch('src.cifar10.micro_child.drop_path', return_value='drop_path')
    @patch('src.cifar10.micro_child.fw.minimum', return_value=0.0)
    @patch('src.cifar10.micro_child.fw.to_float', return_value=1.0)
    def test_apply_drop_path(self, to_float, minimum, drop_path):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
                mc.global_step = 0
                mc._apply_drop_path(None, 0)
                to_float.assert_called_with(310)
                minimum.assert_called_with(1.0, 1.0)
                drop_path.assert_called_with(None, 1.0)

    @patch('src.cifar10.child.Child.FactorizedReduction')
    @patch('src.cifar10.micro_child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    def test_maybe_calibrate_size_same(self, relu, conv2d, batch_norm1, batch_norm2, fr):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
                with patch.object(mc.weights, 'get', return_value="fw.create_weight") as create_weight:
                    layer1 = mock.MagicMock(name="layer1")
                    layer2 = mock.MagicMock(name="layer2")
                    cs = MicroChild.CalibrateSize(mc, [32, 32], [3, 3], 24, True, mc.weights, False)
                    cs([layer1, layer2])
                    create_weight.assert_called_with(False, 'calibrate/pool_y/', 'w', [1, 1, 3, 24], None)
                    relu.assert_called_with(layer2)
                    conv2d.assert_called_with('relu', 'fw.create_weight', [1, 1, 1, 1], 'SAME', data_format='NHWC')
                    batch_norm1.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                    batch_norm1().assert_called_with('conv2d')
                    fr.assert_not_called()

    @patch('src.cifar10.child.Child.FactorizedReduction')
    @patch('src.cifar10.micro_child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    def test_maybe_calibrate_size_different(self, relu, conv2d, batch_norm, fr):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
                with patch.object(mc.weights, "get", return_value="fw.create_weight") as create_weight:
                    layer1 = mock.MagicMock(name="layer1")
                    layer2 = mock.MagicMock(name="layer2")
                    cs = MicroChild.CalibrateSize(mc, [64, 32], [3, 3], 24, True, mc.weights, False)
                    cs([layer1, layer2])
                    create_weight.assert_called_with(False, 'calibrate/pool_y/', 'w', [1, 1, 3, 24], None)
                    relu.assert_called_with(layer2)
                    conv2d.assert_called_with('relu', 'fw.create_weight', [1, 1, 1, 1], 'SAME', data_format="NHWC")
                    batch_norm.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                    batch_norm().assert_called_with('conv2d')
                    fr.assert_called_with(mc, 3, 24, 2, True, mc.weights, False)
                    fr().assert_called_with('relu')

    @patch('src.cifar10.child.Child.FactorizedReduction')
    @patch('src.cifar10.micro_child.MicroChild.ENASLayer', return_value=mock.MagicMock(return_value=mock.MagicMock(return_value='el')))
    @patch('src.cifar10.micro_child.fw.matmul')
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.child.BatchNorm')
    @patch('src.cifar10.micro_child.fw.conv2d', return_value='conv2d')
    @patch('src.cifar10.micro_child.print')
    def test_model_nhwc(self, print, conv2d, batch_norm, relu, matmul, el, fr):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
                mc.name = "MicroChild"
                mc.current_controller_reduce_arc = lambda: None
                mc.current_controller_normal_arc = lambda: None
                mc.keep_prob = None
                el_result = el()
                with patch.object(mc, 'data_format') as data_format:
                    with patch.object(mc.weights, 'get', return_value="fw.create_weight") as create_weight:
                        m = MicroChild.Model(mc, True)
                        m({})
                        create_weight.assert_called_with(False, 'MicroChild/fc/', "w", [96, 10], None)
                        conv2d.assert_called_with({}, "fw.create_weight", [1, 1, 1, 1], 'SAME', data_format=data_format.name)
                        batch_norm.assert_called_with(True, data_format, mc.weights, 72, False)
                        batch_norm().assert_called_with("conv2d")
                        fr.assert_called_with(mc, 48, 96, 2, True, mc.weights, False)
                        fr().assert_called_with(el_result())
                        el.assert_called_with(mc, None, [8, 8], [96, 96], 96, mc.weights, False)
                        el().assert_any_call([el_result(), el_result()])
                        data_format.global_avg_pool.assert_called_with('relu')
                        relu.assert_called_with(el_result())
                        matmul.assert_called_with(data_format.global_avg_pool(), 'fw.create_weight')

    @patch('src.cifar10.child.Child.FactorizedReduction')
    @patch('src.cifar10.micro_child.MicroChild.ENASLayer', return_value=mock.MagicMock(return_value=mock.MagicMock(return_value='el')))
    @patch('src.cifar10.micro_child.fw.matmul')
    @patch('src.cifar10.micro_child.fw.dropout', return_value="dropout")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.child.BatchNorm')
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.print')
    def test_model_nchw(self, print1, conv2d, batch_norm, relu, do, matmul, el, fr):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nchw):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
                with patch.object(mc.data_format, 'global_avg_pool', return_value="gap") as gap:
                    el_value = el()
                    with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                        mc.name = "MicroChild"
                        mc.current_controller_reduce_arc = lambda: None
                        mc.current_controller_normal_arc = lambda: None
                        mc.keep_prob = 0.9
                        m = MicroChild.Model(mc, True)
                        m({})
                        create_weight.assert_any_call(False, 'MicroChild/stem_conv/', "w", [3, 3, 3, 72], None)
                        conv2d.assert_called_with({}, "fw.create_weight", [1, 1, 1, 1], 'SAME', data_format="NCHW")
                        batch_norm.assert_called_with(True, mc.data_format, mc.weights, 72, False)
                        batch_norm().assert_called_with('conv2d')
                        fr.assert_called_with(mc, 48, 96, 2, True, mc.weights, False)
                        fr().assert_called_with(el_value())
                        el.assert_called_with(mc, None, [8, 8], [96, 96], 96, mc.weights, False)
                        el().assert_any_call([el_value(), el_value()])
                        for num in range(4):
                            print1.assert_any_call(f"Layer  {num}: {el_value()}")
                        relu.assert_called_with(el_value())
                        gap.assert_called_with("relu")
                        do.assert_called_with("gap", 0.9)
                        create_weight.assert_called_with(False, 'MicroChild/fc/', "w", [96, 10], None)
                        matmul.assert_called_with('dropout', "fw.create_weight")

    @patch('src.cifar10.child.Child.FactorizedReduction')
    @patch('src.cifar10.micro_child.MicroChild.ENASLayer', return_value=mock.MagicMock(return_value='el'))
    @patch('src.cifar10.micro_child.fw.avg_pool2d')
    @patch('src.cifar10.micro_child.fw.matmul', return_value="matmul")
    @patch('src.cifar10.micro_child.fw.dropout', return_value="dropout")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value='batch_norm'))
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.print')
    def test_model_aux_heads(self, print0, conv2d, batch_norm, relu, do, matmul, avg_pool2d, el, fr):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nchw):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            with patch.object(mc.data_format, 'global_avg_pool') as gap:
                gap.get_shape = mock.MagicMock(return_value=[3, 3])
                with patch.object(mc.weights, "get", return_value="fw.create_weight") as create_weight:
                    mc.name = "MicroChild"
                    mc.current_controller_reduce_arc = lambda: None
                    mc.current_controller_normal_arc = lambda: None
                    mc.keep_prob = 0.9
                    mc.use_aux_heads = True
                    mc.aux_head_indices = [0]
                    m = MicroChild.Model(mc, True)
                    m({})
                    create_weight.assert_any_call(False, 'MicroChild/stem_conv/', "w", [3, 3, 3, 72], None)
                    conv2d.assert_called_with("relu", "fw.create_weight", [1, 1, 1, 1], 'SAME', data_format="NCHW")
                    batch_norm.assert_called_with(True, mc.data_format, mc.weights, 768, False)
                    batch_norm().assert_called_with('conv2d')
                    el.assert_called_with(mc, None, [8, 8], [96, 96], 96, mc.weights, False)
                    fr.assert_called_with(mc, 48, 96, 2, True, mc.weights, False)
                    fr().assert_called_with(el()())
                    for num in range(4):
                        print0.assert_any_call(f"Layer  {num}: el")
                    relu.assert_called_with('el')
                    gap.assert_called_with("relu")
                    do.assert_called_with(gap(), 0.9)
                    create_weight.assert_called_with(False, 'MicroChild/fc/', "w", [96, 10], None)
                    matmul.assert_called_with('dropout', "fw.create_weight")
                    avg_pool2d.assert_called_with([5, 5], [3, 3], "VALID", data_format="channels_first")
                    avg_pool2d().assert_called_with('relu')

    @patch('src.cifar10.micro_child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.micro_child.fw.separable_conv2d', return_value="s_conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    def test_fixed_conv(self, relu, s_conv2d, batch_norm):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
                with patch.object(mc.weights, 'get', return_value="fw.create_weight") as create_weight:
                    fc = MicroChild.FixedConv(mc, 3, 3, 24, 1, True, mc.weights, False)
                    fc(None)
                    create_weight.assert_called_with(False, 'sep_conv_1/', "w_point", [1, 1, 3, 24], None)
                    relu.assert_called_with("batch_norm")
                    s_conv2d.assert_called_with("relu", depthwise_filter="fw.create_weight", pointwise_filter="fw.create_weight", strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
                    batch_norm.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                    batch_norm().assert_called_with('s_conv2d')

    def test_fixed_combine_small_hw(self):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nchw):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
                fc = MicroChild.FixedCombine(mc, [0], [3], [32], 32, 24, True, mc.weights, True)
                fc([1])

    @patch('src.cifar10.child.Child.FactorizedReduction')
    @patch('src.cifar10.micro_child.fw.concat', return_value="concat")
    def test_fixed_combine_large_hw(self, concat, fr):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
                layer1 = mock.MagicMock(name='layer1')
                layer1.get_shape = mock.MagicMock(return_value=[1, 32, 32, 3])
                layer2 = mock.MagicMock(name='layer2')
                layer2.get_shape = mock.MagicMock(return_value=[1, 64, 64, 3])
                fc = MicroChild.FixedCombine(mc, [0, 0], [3, 3], [32, 64], 32, 24, True, mc.weights, True)
                fc([layer1, layer2])
                fr.assert_called_with(mc, 3, 24, 2, True, mc.weights, True)
                fr().assert_called_with(layer2)
                concat.assert_called_with([layer1, fr()()], axis=3)

    @patch('src.cifar10.micro_child.MicroChild.FixedCombine', return_value=mock.MagicMock(return_value='fc'))
    @patch('src.cifar10.micro_child.MicroChild.CalibrateSize')
    @patch('src.cifar10.micro_child.np.zeros', return_value=np.zeros([7], dtype=np.int32))
    @patch('src.cifar10.micro_child.fw.max_pool2d')
    @patch('src.cifar10.micro_child.fw.avg_pool2d')
    @patch('src.cifar10.micro_child.fw.separable_conv2d', return_value="s_conv2d")
    @patch('src.cifar10.child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm1"))
    @patch('src.cifar10.micro_child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm2"))
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    def test_fixed_layer(self, relu, conv2d, batch_norm2, batch_norm1, s_conv2d, avg_pool2d, max_pool2d, np_zeros, mcs, fc):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
                with patch.object(mc, '_apply_drop_path', return_value="adp") as adp:
                    with patch.object(mc.weights, 'get', return_value="fw.create_weight") as create_weight:
                        layer1 = mock.MagicMock(name='layer1')
                        layer2 = mock.MagicMock(name='layer2')
                        mcs().return_value = [0, 0]
                        fl = MicroChild.FixedLayer(mc, 0, [0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 2, 0, 3, 0, 3, 0, 4, 0, 4], [32, 32], [3, 3], 24, 1, True, mc.weights, False)
                        self.assertEqual('fc', fl([layer1, layer2]))
                        self.assertEqual(144, fl.out_chan)
                        mcs.assert_called_with(mc, [32, 32], [3, 3], 24, True, mc.weights, False)
                        mcs().assert_called_with([layer1, layer2])
                        create_weight.assert_called_with(False, 'cell_1/y_conv/sep_conv_1/', "w_point", [1, 1, 24, 24], None)
                        relu.assert_called_with('batch_norm2')
                        conv2d.assert_called_with("relu", "fw.create_weight", [1, 1, 1, 1], "SAME", data_format="NHWC")
                        batch_norm2.assert_called_with(True, mc.data_format, mc.weights, 24, False)
                        batch_norm2().assert_called_with('s_conv2d')
                        s_conv2d.assert_called_with("relu", depthwise_filter="fw.create_weight", pointwise_filter="fw.create_weight", strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
                        adp.assert_called_with(max_pool2d()(), 0)
                        fc.assert_called_with(mc, np_zeros(), [24] * 7, [16] * 7, 16, 24, True, mc.weights, False)
                        fc().assert_called_with([0, 'batch_norm2', 'adpadp', 'adpadp', 'adpadp', 'adpadp', 0])

    @patch('src.cifar10.micro_child.MicroChild.ENASConvOuter')
    @patch('src.cifar10.micro_child.fw.stack', return_value=tf.constant(np.ndarray((1, 4, 32, 32, 3))))
    @patch('src.cifar10.micro_child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.micro_child.fw.max_pool2d', return_value=mock.MagicMock(return_value='max_pool2d'))
    @patch('src.cifar10.micro_child.fw.avg_pool2d', return_value=mock.MagicMock(return_value="avg_pool2d"))
    def test_enas_cell(self, avg_pool2d, max_pool2d, reshape, relu, conv2d, batch_norm, stack, ec):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            mc = MicroChild({}, {})
            with patch.object(mc.weights, 'get', return_value="fw.create_weight") as create_weight:
                mcec = MicroChild.ENASCell(mc, 0, 1, 3, 24, mc.weights, False)
                mcec(None, 0)
                avg_pool2d.assert_called_with([3, 3], [1, 1], "SAME", data_format="channels_last")
                avg_pool2d().assert_called_with(None)
                create_weight.assert_called_with(False, 'x_conv/', "w", [1, 72], None)
                reshape.assert_called_with('w', [1, 1, 3, 24])
                relu.assert_called_with(None)
                conv2d.assert_called_with("relu", 'reshape', strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
                batch_norm.assert_called_with(True, mc.data_format, mc.weights, mc.out_filters, False)
                batch_norm().assert_called_with('conv2d')
                ec.assert_called_with(mc, 0, 1, 5, 24, mc.weights, False)
                ec().assert_called_with('batch_norm')
                stack.assert_called_with([ec()(), ec()(), 'batch_norm', 'batch_norm', 'batch_norm'], axis=0)

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
                with patch.object(mc.weights, 'get') as create_weight:
                    create_weight().__getitem__ = mock.MagicMock()
                    ec = MicroChild.ENASConvOuter(mc, 0, 0, 3, 24, mc.weights, False)
                    ec(None)
                    create_weight.assert_any_call(False, 'conv_3x3/stack_1/bn/', 'scale', [2, 24], 'ones')
                    create_weight.assert_called_with(False, 'conv_3x3/stack_1/', 'w_point', [2, 576], None)
                    reshape.assert_called_with((), (0,))
                    relu.assert_called_with('f')
                    s_conv.assert_called_with('relu', depthwise_filter='reshape', pointwise_filter='reshape', strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
                    fbn.assert_called_with(x='s_conv2d', scale=create_weight().__getitem__(), offset=create_weight().__getitem__(), mean='reshape', variance='reshape', epsilon=1e-05, data_format='NHWC', is_training=True)
                    zeros.assert_called_with()
                    ones.assert_called_with()

    @patch('src.cifar10.micro_child.MicroChild.ENASCell', return_value=mock.MagicMock(name='first_call', return_value='ec'))
    @patch('src.cifar10.micro_child.MicroChild.CalibrateSize')
    @patch('src.cifar10.micro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.micro_child.fw.where', return_value="where")
    @patch('src.cifar10.micro_child.fw.add_n', return_value="add_n")
    @patch('src.cifar10.micro_child.fw.one_hot', return_value="one_hot")
    @patch('src.cifar10.micro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.micro_child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.fw.shape', return_value="shape")
    @patch('src.cifar10.micro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.micro_child.fw.transpose', return_value="transpose")
    @patch('src.cifar10.micro_child.fw.gather', return_value="gather")
    @patch('src.cifar10.micro_child.fw.stack') # tf.constant(np.ndarray((2, 3, 32, 32, 3))))
    def test_enas_layer_nhwc(self, stack, gather, transpose, reshape, shape, relu, conv2d, batch_norm, to_int32, one_hot, add_n, where, equal, mcs, ec):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
                mcs().return_value = ['mcs1', 'mcs2']
                with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                    val = stack().__getitem__()
                    prev_layer1 = mock.MagicMock(name="prev_layer1")
                    prev_layer2 = mock.MagicMock(name="prev_layer2")
                    el = MicroChild.ENASLayer(mc, [0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 2, 0, 3, 0, 3, 0, 4, 0, 4], [32, 32], [3, 3], 3, mc.weights, False)
                    el([prev_layer1, prev_layer2])
                    mcs.assert_called_with(mc, [32, 32], [3, 3], 3, True, mc.weights, False)
                    mcs().assert_called_with([prev_layer1, prev_layer2])
                    ec.assert_called_with(mc, 4, 0, 3, 3, mc.weights, False)
                    ec().assert_called_with(val, 4)
                    stack.assert_called_with(['mcs1', 'mcs2', 'ecec', 'ecec', 'ecec', 'ecec', 'ecec'], axis=0)
                    gather.assert_called_with('fw.create_weight', 'reshape', axis=0)
                    gather.assert_any_call(stack(), 'reshape', axis=0)
                    transpose.assert_called_with('gather', [1, 2, 3, 0, 4])
                    reshape.assert_called_with('batch_norm', 'shape')
                    relu.assert_called_with('reshape')
                    conv2d.assert_called_with('relu', 'reshape', strides=[1, 1, 1, 1], padding='SAME', data_format="NHWC")
                    batch_norm.assert_called_with(True, mc.data_format, mc.weights, 3, False)
                    batch_norm().assert_called_with('conv2d')
                    create_weight.assert_called_with(False, 'final_conv/', 'w', [7, 9], None)
                    to_int32.assert_called_with('where')
                    one_hot.assert_called_with(0, depth=7, dtype=tf.int32)
                    add_n.assert_called_with(['one_hot'] * 10)
                    equal.assert_called_with('add_n', 0)
                    where.assert_called_with('equal')

    @patch('src.cifar10.micro_child.MicroChild.ENASCell', return_value=mock.MagicMock(name='first_call', return_value='ec'))
    @patch('src.cifar10.micro_child.MicroChild.CalibrateSize')
    @patch('src.cifar10.micro_child.fw.one_hot', return_value="one_hot")
    @patch('src.cifar10.micro_child.fw.add_n', return_value="add_n")
    @patch('src.cifar10.micro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.micro_child.fw.where', return_value="where")
    @patch('src.cifar10.micro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.micro_child.BatchNorm', return_value=mock.MagicMock(return_value="batch_norm"))
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.fw.shape', return_value="shape")
    @patch('src.cifar10.micro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.micro_child.fw.transpose', return_value="transpose")
    @patch('src.cifar10.micro_child.fw.gather', return_value="gather")
    @patch('src.cifar10.micro_child.fw.stack')
    def test_enas_layer_nchw(self, stack, gather, transpose, reshape, shape, relu, conv2d, batch_norm, to_int32, where, equal, add_n, one_hot, mcs, ec):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nchw):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
                layer1 = mock.MagicMock(name='layer1')
                layer2 = mock.MagicMock(name='layer2')
                mcs().return_value = [layer1, layer2]
                with patch.object(mc.weights, 'get', return_value='fw.create_weight') as create_weight:
                    val = stack().__getitem__()
                    el = MicroChild.ENASLayer(mc, [0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 2, 0, 3, 0, 3, 0, 4, 0, 4], [32, 32], [3, 3], 3, mc.weights, False)
                    el([layer1, layer2])
                    mcs.assert_called_with(mc, [32, 32], [3, 3], 3, True, mc.weights, False)
                    mcs().assert_called_with([layer1, layer2])
                    ec.assert_called_with(mc, 4, 0, 3, 3, mc.weights, False)
                    ec().assert_called_with(val, 4)
                    create_weight.assert_called_with(False, 'final_conv/', 'w', [7, 9], None)
                    stack.assert_called_with([layer1, layer2, 'ecec', 'ecec', 'ecec', 'ecec', 'ecec'], axis=0)
                    gather.assert_called_with('fw.create_weight', 'reshape', axis=0)
                    gather.assert_any_call(stack(), 'reshape', axis=0)
                    transpose.assert_called_with('gather', [1, 0, 2, 3, 4])
                    shape.assert_called_with(val)
                    reshape.assert_called_with('batch_norm', 'shape')
                    relu.assert_called_with('reshape')
                    conv2d.assert_called_with('relu', 'reshape', strides=[1, 1, 1, 1], padding='SAME', data_format="NCHW")
                    batch_norm.assert_called_with(True, mc.data_format, mc.weights, 3, False)
                    batch_norm().assert_called_with('conv2d')
                    to_int32.assert_called_with('where')
                    one_hot.assert_called_with(0, depth=7, dtype=tf.int32)
                    add_n.assert_called_with(['one_hot'] * 10)
                    equal.assert_called_with('add_n', 0)
                    where.assert_called_with('equal')

    @patch('src.cifar10.micro_child.MicroChild.Model', return_value=mock.MagicMock(return_value=('logit', 'aux_logit')))
    @patch('src.cifar10.micro_child.get_train_ops')
    @patch('src.cifar10.micro_child.fw.reduce_sum', return_value=2.0)
    @patch('src.cifar10.micro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.micro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.micro_child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.micro_child.fw.reduce_mean', return_value=1.0)
    @patch('src.cifar10.micro_child.fw.sparse_softmax_cross_entropy_with_logits', return_value="sscewl")
    @patch('src.cifar10.micro_child.print')
    def test_build_train_aux_heads(self, print, sscewl, reduce_mean, argmax, to_int32, equal, reduce_sum, get_train_ops, model):
        train_op = mock.MagicMock(name="get_train_op", return_value='train_op')
        grad_norm = mock.MagicMock(name="get_grad_norm", return_value='grad_norm')
        get_train_ops.return_value = (train_op, 2, grad_norm, 4)
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
                mc.name = "MicroChild"
                mc.use_aux_heads = True
                dataset_iter = mock.MagicMock()
                dataset_iter.__next__ = mock.MagicMock(return_value=('images', 'labels'))
                dataset = mock.MagicMock()
                dataset.as_numpy_iterator = mock.MagicMock(return_value=dataset_iter)
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
                loss0, train_loss0, train_acc0, train_op0, lr, grad_norm0, optimizer = mc._build_train(dataset)
                logits_aux_logits = MicroChild.Model(mc, True)(dataset)
                train_loss = train_loss0(logits_aux_logits)
                loss = loss0(logits_aux_logits)
                train_acc = train_acc0(logits_aux_logits)
                train_op = train_op0(train_loss, mc.trainable_variables())
                grad_norm = grad_norm0(train_loss, mc.trainable_variables())
                self.assertEqual(loss, 1.0)
                self.assertEqual(1.4, train_loss)
                self.assertEqual(2.0, train_acc)
                self.assertEqual('train_op', train_op)
                self.assertEqual(2, lr)
                self.assertEqual('grad_norm', grad_norm)
                self.assertEqual(4, optimizer)
                print.assert_any_call('-' * 80)
                print.assert_any_call("Build train graph")
                model.assert_called_with(mc, True)
                model().assert_called_with(dataset)
                sscewl.assert_called_with(logits='logit', labels=dataset_iter.__next__().__getitem__(1))
                reduce_mean.assert_called_with("sscewl")
                argmax.assert_called_with('logit', axis=1)
                to_int32.assert_called_with("equal")
                equal.assert_called_with("to_int32", 'labels')
                reduce_sum.assert_called_with("to_int32")
                get_train_ops.assert_called_with(mc.global_step, mc.learning_rate, clip_mode=None, l2_reg=None, num_train_batches=1, optim_algo=None)

    @patch('src.cifar10.micro_child.MicroChild.Model', return_value=mock.MagicMock(return_value=('logit', 'aux_logit')))
    @patch('src.cifar10.micro_child.get_train_ops')
    @patch('src.cifar10.micro_child.fw.reduce_sum', return_value=2.0)
    @patch('src.cifar10.micro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.micro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.micro_child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.micro_child.fw.reduce_mean', return_value=1.0)
    @patch('src.cifar10.micro_child.fw.sparse_softmax_cross_entropy_with_logits', return_value="sscewl")
    @patch('src.cifar10.micro_child.print')
    def test_build_train_no_aux_heads(self, print, sscewl, reduce_mean, argmax, to_int32, equal, reduce_sum, get_train_ops, model):
        train_op = mock.MagicMock(name="get_train_op", return_value='grad_norm')
        grad_norm = mock.MagicMock(name="get_grad_norm", return_value='grad_norm')
        get_train_ops.return_value = (train_op, 2, grad_norm, 4)
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
                mc.name = "MicroChild"
                mc.use_aux_heads = False
                dataset_iter = mock.MagicMock()
                dataset_iter.__next__ = mock.MagicMock(return_value=('images', 'labels'))
                dataset = mock.MagicMock()
                dataset.as_numpy_iterator = mock.MagicMock(return_value=dataset_iter)
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
                loss0, train_loss0, train_acc0, train_op0, lr, grad_norm0, optimizer = mc._build_train(dataset)
                logits_aux_logits = MicroChild.Model(mc, True)(dataset)
                train_loss = train_loss0(logits_aux_logits)
                loss = loss0(logits_aux_logits)
                train_acc = train_acc0(logits_aux_logits)
                train_op = train_op0(train_loss, mc.trainable_variables())
                grad_norm = grad_norm0(train_loss, mc.trainable_variables())
                self.assertEqual(loss, 1.0)
                self.assertEqual(1.0, train_loss)
                self.assertEqual(2.0, train_acc)
                self.assertEqual(2, lr)
                self.assertEqual('grad_norm', grad_norm)
                print.assert_any_call('-' * 80)
                print.assert_any_call("Build train graph")
                model.assert_called_with(mc, True)
                model().assert_called_with(dataset)
                sscewl.assert_called_with(logits='logit', labels=dataset_iter.__next__().__getitem__(1))
                reduce_mean.assert_called_with("sscewl")
                argmax.assert_called_with('logit', axis=1)
                to_int32.assert_called_with("equal")
                equal.assert_called_with("to_int32", 'labels')
                reduce_sum.assert_called_with("to_int32")
                get_train_ops.assert_called_with(mc.global_step, mc.learning_rate, clip_mode=None, l2_reg=None, num_train_batches=1, optim_algo=None)

    @patch('src.cifar10.micro_child.MicroChild.Model', return_value=mock.MagicMock(return_value='model'))
    @patch('src.cifar10.micro_child.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.micro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.micro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.micro_child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.micro_child.print')
    def test_build_valid(self, print, argmax, to_int32, equal, reduce_sum, model):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            mc.x_valid = {}
            mc.y_valid = {}
            logits = MicroChild.Model(mc, False, True)(mc.x_valid)
            predictions, accuracy = mc._build_valid(mc.y_valid)
            self.assertEqual(('to_int32', 'reduce_sum'), (predictions(logits), accuracy(logits)))
            print.assert_any_call('-' * 80)
            print.assert_any_call("Build valid graph")
            argmax.assert_called_with("model", axis=1)
            to_int32.assert_any_call("argmax")
            equal.assert_any_call("to_int32", {})
            to_int32.assert_any_call("equal")
            reduce_sum.assert_called_with("to_int32")
            model.assert_called_with(mc, False, True)
            model().assert_called_with({})

    @patch('src.cifar10.micro_child.MicroChild.Model', return_value=mock.MagicMock(return_value='model'))
    @patch('src.cifar10.micro_child.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.micro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.micro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.micro_child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.micro_child.print')
    def test_build_test(self, print, argmax, to_int32, equal, reduce_sum, model):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
            mc.x_test = {}
            mc.y_test = {}
            logits = MicroChild.Model(mc, False, True)(mc.x_test)
            predictions, accuracy = mc._build_test(mc.y_test)
            self.assertEqual(('to_int32', 'reduce_sum'), (predictions(logits), accuracy(logits)))
            print.assert_any_call('-' * 80)
            print.assert_any_call("Build test graph")
            argmax.assert_called_with("model", axis=1)
            to_int32.assert_any_call("argmax")
            equal.assert_any_call("to_int32", {})
            to_int32.assert_any_call("equal")
            reduce_sum.assert_called_with("to_int32")
            model.assert_called_with(mc, False, True)
            model().assert_called_with({})

    @patch('src.cifar10.child.Child.__init__')
    @patch('src.cifar10.micro_child.MicroChild.Model', return_value=mock.MagicMock(return_value='model'))
    @patch('src.cifar10.micro_child.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.micro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.micro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.micro_child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.micro_child.fw.shuffle_batch')
    @patch('src.cifar10.micro_child.print')
    def test_build_valid_rl(self, print, shuffle_batch, argmax, to_int32, equal, reduce_sum, model, _Child):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            mc = MicroChild({}, {})
            mc.images = { 'valid_original': np.ndarray((5000, 32, 32, 3)) }
            mc.labels = { 'valid_original': np.ndarray((5000)) }
            mc.batch_size = 32
            mc.seed = None
            shuffle_batch.return_value = fw.shuffle_batch((mc.images, mc.labels), mc.batch_size, mc.seed, 25000)
            shuffle = MicroChild.ValidationRLShuffle(mc, True)
            vrl = MicroChild.ValidationRL()
            dataset_valid_shuffle = shuffle(mc.images['valid_original'], mc.labels['valid_original'])
            logits = MicroChild.Model(mc, True, True)(dataset_valid_shuffle)
            vrl(logits, dataset_valid_shuffle)
            shuffle_batch.assert_called_with(
                (mc.images['valid_original'], mc.labels['valid_original']),
                mc.batch_size,
                mc.seed,
                25000)
            argmax.assert_called_with('model', axis=1)
            to_int32.assert_any_call("argmax")
            equal.assert_called_with("to_int32", shuffle_batch().map())
            to_int32.assert_any_call("equal")
            reduce_sum.assert_called_with("to_int32")

    def test_connect_controller(self):
        with patch('src.cifar10.micro_child.Child.__init__', new=mock_init_nhwc):
            with tf.Graph().as_default():
                mc = MicroChild({}, {})
                with patch.object(mc, '_build_train', return_value=('loss', 'train_loss', 'train_acc', 'train_op', 'lr', 'grad_norm', 'optimizer')) as build_train:
                    with patch.object(mc, '_build_valid', return_value=('predictions', 'accuracy')) as build_valid:
                        with patch.object(mc, '_build_test', return_value=('predictions', 'accuracy')) as build_test:
                            mock_controller = mock.MagicMock()
                            mc.connect_controller(mock_controller)
                            build_train.assert_called_with(mc.dataset)
                            build_valid.assert_called_with(mc.dataset_valid)
                            build_test.assert_called_with(mc.dataset_test)

if "__main__" == "__name__":
    unittest.main()
