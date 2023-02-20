import unittest
import unittest.mock as mock
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from src.cifar10.micro_child import MicroChild

def mock_init(self, images, labels, **kwargs):
    self.data_format = kwargs['data_format']
    self.num_train_batches = 1


class TestMicroChild(unittest.TestCase):
    def test_micro_child_nhwc(self):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            self.assertEqual(MicroChild, type(mc))

    def test_micro_child_nchw(self):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, data_format="NCHW", use_aux_heads=True)
            self.assertEqual(MicroChild, type(mc))

    def test_micro_child_raises(self):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            self.assertRaises(ValueError, MicroChild, {}, {}, num_epochs=310, data_format="INVALID")

    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.concat', return_value="concat")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.create_weight', return_value="fw.create_weight")
    @patch('src.cifar10.micro_child.fw.avg_pool', return_value="avg_pool")
    @patch('src.cifar10.micro_child.fw.pad', return_value=tf.constant(np.ones((4, 32, 32, 3))))
    def test_factorized_reduction_nhwc_stride2(self, pad, avg_pool, create_weight, conv2d, concat, batch_norm):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            with patch.object(mc, "_get_C", return_value=3) as getc:
                with patch.object(mc, "_get_strides", return_value="get_strides") as gets:
                    retval = mc._factorized_reduction(None, 24, 2, True)
                    gets.assert_called_with(2)
                    pad.assert_called_with(None, [[0, 0], [0, 1], [0, 1], [0, 0]])
                    avg_pool.assert_called()
                    getc.assert_called_with('avg_pool')
                    create_weight.assert_called_with('w', [1, 1, 3, 12])
                    conv2d.assert_called_with('avg_pool', 'fw.create_weight', [1, 1, 1, 1], "VALID", data_format="NHWC")
                    concat.assert_called_with(values=['conv2d', 'conv2d'], axis=3)
                    batch_norm.assert_called_with("concat", True, data_format="NHWC")
                    self.assertEqual("batch_norm", retval)

    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.concat', return_value="concat")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.create_weight', return_value="fw.create_weight")
    @patch('src.cifar10.micro_child.fw.avg_pool', return_value="avg_pool")
    @patch('src.cifar10.micro_child.fw.pad', return_value=tf.constant(np.ones((4, 32, 32, 3))))
    def test_factorized_reduction_nchw_stride2(self, pad, avg_pool, create_weight, conv2d, concat, batch_norm):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            mc.data_format = "NCHW"
            with patch.object(mc, "_get_C", return_value=3) as getc:
                with patch.object(mc, "_get_strides", return_value="get_strides") as gets:
                    retval = mc._factorized_reduction(None, 24, 2, True)
                    gets.assert_called_with(2)
                    pad.assert_called_with(None, [[0, 0], [0, 0], [0, 1], [0, 1]])
                    avg_pool.assert_called()
                    getc.assert_called_with('avg_pool')
                    create_weight.assert_called_with('w', [1, 1, 3, 12])
                    conv2d.assert_called_with('avg_pool', 'fw.create_weight', [1, 1, 1, 1], "VALID", data_format="NCHW")
                    concat.assert_called_with(values=['conv2d', 'conv2d'], axis=1)
                    batch_norm.assert_called_with("concat", True, data_format="NCHW")
                    self.assertEqual("batch_norm", retval)

    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.create_weight', return_value="fw.create_weight")
    def test_factorized_reduction_nchw_stride1(self, create_weight, conv2d, batch_norm):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            mc.data_format = "NCHW"
            with patch.object(mc, "_get_C", return_value=3) as getc:
                with patch.object(mc, "_get_strides", return_value="get_strides") as gets:
                    retval = mc._factorized_reduction(None, 24, 1, True)
                    create_weight.assert_called_with("w", [1, 1, 3, 24])
                    conv2d.assert_called_with(None, "fw.create_weight", [1, 1, 1, 1], 'SAME', data_format="NCHW")
                    batch_norm.assert_called_with('conv2d', True, data_format="NCHW")

    def test_get_c_nchw(self):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            mc.data_format = "NCHW"
            self.assertEqual(3, mc._get_C(tf.constant(np.ndarray((45000, 3, 32, 32)))))

    def test_get_c_nhwc(self):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            mc.data_format = "NHWC"
            self.assertEqual(3, mc._get_C(tf.constant(np.ndarray((45000, 32, 32, 3)))))

    def test_get_c_raises(self):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            mc.data_format = "INVALID"
            self.assertRaises(ValueError, mc._get_C, tf.constant(np.ndarray((4, 32, 32, 3))))

    def test_get_hw(self):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            self.assertEqual(mc._get_HW(tf.constant(np.ndarray((1, 2, 3)))), 3)

    def test_get_strides_nhwc(self):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            mc.data_format = "NHWC"
            self.assertEqual([1, 2, 2, 1], mc._get_strides(2))

    def test_get_strides_nchw(self):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            mc.data_format = "NCHW"
            self.assertEqual([1, 1, 2, 2], mc._get_strides(2))

    def test_get_strides_raises(self):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            mc.data_format = "INVALID"
            self.assertRaises(ValueError, mc._get_strides, 2)

    @patch('src.cifar10.macro_controller.fw.to_float', return_value=1.0)
    @patch('src.cifar10.macro_controller.fw.divide', return_value=1.0)
    @patch('src.cifar10.macro_controller.fw.shape', return_value=[5])
    def test_apply_drop_path(self, shape, divide, to_float):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            mc._apply_drop_path(None, 0)
            shape.assert_called_with(None)
            divide.assert_called()
            to_float.assert_called_with(310)

    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.fw.create_weight', return_value="fw.create_weight")
    def test_maybe_calibrate_size_same(self, create_weight, relu, conv2d, batch_norm):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            mc._maybe_calibrate_size([
                tf.constant(np.ndarray((45000, 32, 32, 3))),
                tf.constant(np.ndarray((45000, 32, 32, 3)))], 24, True)
            create_weight.assert_called_with('w', [1, 1, 3, 24])
            relu.assert_called()
            conv2d.assert_called_with('relu', 'fw.create_weight', [1, 1, 1, 1], 'SAME', data_format='NHWC')
            batch_norm.assert_called_with('conv2d', True, data_format='NHWC')

    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.fw.create_weight', return_value="fw.create_weight")
    def test_maybe_calibrate_size_different(self, create_weight, relu, conv2d, batch_norm):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            with patch.object(mc, '_factorized_reduction', return_value="fr"):
                mc._maybe_calibrate_size([
                    tf.constant(np.ndarray((45000, 16, 32, 3))),
                    tf.constant(np.ndarray((45000, 32, 16, 3)))], 24, True)
                create_weight.assert_called_with('w', [1, 1, 3, 24])
                relu.assert_called()
                conv2d.assert_called_with('relu', 'fw.create_weight', [1, 1, 1, 1], 'SAME', data_format="NHWC")
                batch_norm.assert_called_with('conv2d', True, data_format="NHWC")

    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.create_weight', return_value="fw.create_weight")
    def test_model_nhwc_KNOWN_TO_FAIL(self, create_weight, conv2d, batch_norm):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            mc.name = "MicroChild"
            mc._model({}, True)
            create_weight.assert_called_with("w", [3, 3, 3, 72])
            conv2d.assert_called_with({}, "fw.create_weight", [1, 1, 1, 1], 'SAME', data_format="NHWC")
            batch_norm.assert_called_with("conv2d", True, data_format="NHWC")

    @patch('src.cifar10.micro_child.fw.matmul', return_value="matmul")
    @patch('src.cifar10.micro_child.fw.dropout', return_value="dropout")
    @patch('src.cifar10.micro_child.global_avg_pool', return_value="gap")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.create_weight', return_value="fw.create_weight")
    @patch('src.cifar10.micro_child.print')
    def test_model_nchw(self, print, create_weight, conv2d, batch_norm, relu, gap, do, matmul):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            with patch.object(mc, '_factorized_reduction', return_value="fr") as fr:
                with patch.object(mc, '_enas_layer', return_value="el") as el:
                    with patch.object(mc, '_get_C', return_value="get_c") as get_c:
                        mc.name = "MicroChild"
                        mc.data_format = "NCHW"
                        mc.reduce_arc = None
                        mc.normal_arc = None
                        mc.keep_prob = 0.9
                        mc._model({}, True)
                        create_weight.assert_any_call("w", [3, 3, 3, 72])
                        conv2d.assert_called_with({}, "fw.create_weight", [1, 1, 1, 1], 'SAME', data_format="NCHW")
                        batch_norm.assert_called_with("conv2d", True, data_format="NCHW")
                        fr.assert_called_with('el', 96, 2, True)
                        el.assert_called_with(3, ['el', 'el'], None, 96)
                        for num in range(4):
                            print.assert_any_call(f"Layer  {num}: el")
                        relu.assert_called_with('el')
                        gap.assert_called_with("relu", data_format="NCHW")
                        do.assert_called_with("gap", 0.9)
                        get_c.assert_called_with('dropout')
                        create_weight.assert_called_with("w", ["get_c", 10])
                        matmul.assert_called_with('dropout', "fw.create_weight")

    @patch('src.cifar10.micro_child.fw.avg_pool2d')
    @patch('src.cifar10.micro_child.fw.matmul', return_value="matmul")
    @patch('src.cifar10.micro_child.fw.dropout', return_value="dropout")
    @patch('src.cifar10.micro_child.global_avg_pool', return_value=tf.constant(np.ndarray((3, 3, 32, 32))))
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.create_weight', return_value="fw.create_weight")
    @patch('src.cifar10.micro_child.print')
    def test_model_aux_heads(self, print, create_weight, conv2d, batch_norm, relu, gap, do, matmul, avg_pool2d):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            with patch.object(mc, '_factorized_reduction', return_value="fr") as fr:
                with patch.object(mc, '_enas_layer', return_value="el") as el:
                    with patch.object(mc, '_get_C', return_value="get_c") as get_c:
                        with patch.object(mc, "_get_HW", return_value="get_hw") as get_hw:
                            mc.name = "MicroChild"
                            mc.data_format="NCHW"
                            mc.reduce_arc = None
                            mc.normal_arc = None
                            mc.keep_prob = 0.9
                            mc.use_aux_heads = True
                            mc.aux_head_indices = [0]
                            mc._model({}, True)
                            create_weight.assert_any_call("w", [3, 3, 3, 72])
                            conv2d.assert_called_with("relu", "fw.create_weight", [1, 1, 1, 1], 'SAME', data_format="NCHW")
                            batch_norm.assert_called_with("conv2d", is_training=True, data_format="NCHW")
                            fr.assert_called_with('el', 96, 2, True)
                            el.assert_called_with(3, ['el', 'el'], None, 96)
                            for num in range(4):
                                print.assert_any_call(f"Layer  {num}: el")
                            relu.assert_called_with('el')
                            gap.assert_called_with("relu", data_format="NCHW")
                            do.assert_called()
                            get_c.assert_called_with('dropout')
                            create_weight.assert_called_with("w", ["get_c", 10])
                            matmul.assert_called_with('dropout', "fw.create_weight")
                            avg_pool2d.assert_called_with("relu", [5, 5], [3, 3], "VALID", data_format="channels_last")
                            get_hw.assert_called_with("relu")

    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.separable_conv2d', return_value="s_conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.fw.create_weight', return_value="fw.create_weight")
    def test_fixed_conv(self, create_weight, relu, s_conv2d, batch_norm):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            with patch.object(mc, '_get_C', return_value="get_c") as get_c:
                mc._fixed_conv(None, 3, 24, 1, True)
                get_c.assert_called_with('batch_norm')
                # create_weight.assert_any_call("w_depth", [3, 3, "get_c", 1])
                create_weight.assert_called_with("w_point", [1, 1, "get_c", 24])
                relu.assert_called_with("batch_norm")
                s_conv2d.assert_called_with("relu", depthwise_filter="fw.create_weight", pointwise_filter="fw.create_weight", strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
                batch_norm.assert_called_with("s_conv2d", True, data_format="NHWC")

    def test_fixed_combine_small_hw(self):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            mc.data_format = "NCHW"
            with patch.object(mc, '_get_HW', return_value=32) as get_HW:
                mc._fixed_combine([1], [0], 24, True)
                get_HW.assert_called_with(1)

    @patch('src.cifar10.micro_child.fw.concat', return_value="concat")
    def test_fixed_combine_large_hw(self, concat):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            mc.data_format = "NHWC"
            with patch.object(mc, '_factorized_reduction', return_value='fr') as fr:
                mc._fixed_combine([tf.constant(np.ndarray((1, 32, 32, 3))), tf.constant(np.ndarray((1, 64, 64, 3)))], [0, 0], 24, True)
                fr.assert_called()
                concat.assert_called()

    def test_fixed_combine_raises(self):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            mc.data_format = "INVALID"
            with patch.object(mc, '_get_HW', return_value=32) as get_HW:
                self.assertRaises(ValueError, mc._fixed_combine, [1], [0], 24, True)

    @patch('src.cifar10.micro_child.fw.max_pool2d')
    @patch('src.cifar10.micro_child.fw.avg_pool2d')
    @patch('src.cifar10.micro_child.fw.separable_conv2d', return_value="s_conv2d")
    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.fw.create_weight', return_value="fw.create_weight")
    def test_fixed_layer(self, create_weight, relu, conv2d, batch_norm, s_conv2d, avg_pool2d, max_pool2d):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            with patch.object(mc, '_maybe_calibrate_size', return_value=[0, 1]) as mcs:
                with patch.object(mc, '_get_C', return_value="get_c") as get_c:
                    with patch.object(mc, '_apply_drop_path', return_value="adp") as adp:
                        with patch.object(mc, '_fixed_combine', return_value="fc") as fc:
                            self.assertEqual("fc", mc._fixed_layer(0, [tf.constant(np.ndarray((1, 32, 32, 3)))] * 2, [0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 2, 0, 3, 0, 3, 0, 4, 0, 4], 24, 1, True))
                            mcs.assert_called()
                            get_c.assert_called_with(0)
                            create_weight.assert_called_with("w", [1, 1, 'get_c', 24])
                            relu.assert_called_with(0)
                            conv2d.assert_called_with("relu", "fw.create_weight", [1, 1, 1, 1], "SAME", data_format="NHWC")
                            batch_norm.assert_called_with("conv2d", True, data_format="NHWC")
                            s_conv2d.assert_called_with("relu", depthwise_filter="fw.create_weight", pointwise_filter="fw.create_weight", strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
                            adp.assert_called_with('batch_norm', 0)
                            fc.assert_called()

    @patch('src.cifar10.micro_child.fw.stack', return_value=tf.constant(np.ndarray((1, 4, 32, 32, 3))))
    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.micro_child.fw.create_weight', return_value="fw.create_weight")
    @patch('src.cifar10.micro_child.fw.max_pool2d', return_value="max_pool2d")
    @patch('src.cifar10.micro_child.fw.avg_pool2d', return_value="avg_pool2d")
    def test_enas_cell(self, avg_pool2d, max_pool2d, create_weight, reshape, relu, conv2d, batch_norm, stack):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            with patch.object(mc, '_get_C', return_value="get_c") as get_c:
                with patch.object(mc, '_enas_conv', return_value="ec") as ec:
                    mc._enas_cell(None, 0, 1, 0, 24)
                    avg_pool2d.assert_called_with(None, [3, 3], [1, 1], "SAME", data_format="channels_last")
                    get_c.assert_called_with(None)
                    create_weight.assert_called_with("w", [1, 'get_c' * 24])
                    reshape.assert_called_with('w', [1, 1, 'get_c', 24])
                    relu.assert_called_with(None)
                    conv2d.assert_called_with("relu", 'reshape', strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
                    batch_norm.assert_called_with("conv2d", is_training=True, data_format="NHWC")
                    ec.assert_called_with('batch_norm', 0, 1, 5, 24)
                    stack.assert_called_with(['ec', 'ec', 'batch_norm', 'batch_norm', 'batch_norm'], axis=0)

    @patch('src.cifar10.micro_child.fw.ones_init', return_value="ones")
    @patch('src.cifar10.micro_child.fw.zeros_init', return_value="zeros")
    @patch('src.cifar10.micro_child.fw.fused_batch_norm', return_value="fbn")
    @patch('src.cifar10.micro_child.fw.separable_conv2d', return_value="s_conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.micro_child.fw.create_weight', return_value=tf.constant(np.ndarray((3, 3))))
    def test_enas_conv(self, create_weight, reshape, relu, s_conv, fbn, zeros, ones):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            with patch.object(mc, '_get_C', return_value="get_c") as get_c:
                mc._enas_conv(None, 0, 0, 3, 24)
                get_c.assert_called_with('f')
                create_weight.assert_called()
                reshape.assert_called()
                relu.assert_called_with('f')
                s_conv.assert_called_with('relu', depthwise_filter='reshape', pointwise_filter='reshape', strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
                fbn.assert_called()
                zeros.assert_called_with()
                ones.assert_called_with()

    @patch('src.cifar10.micro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.micro_child.fw.create_weight', return_value="fw.create_weight")
    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.micro_child.fw.transpose', return_value="transpose")
    @patch('src.cifar10.micro_child.fw.gather', return_value="gather")
    @patch('src.cifar10.micro_child.fw.stack', return_value=tf.constant(np.ndarray((2, 3, 32, 32, 3))))
    def test_enas_layer_nhwc(self, stack, gather, transpose, reshape, relu, conv2d, batch_norm, create_weight, to_int32):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            mc.data_format = "NHWC"
            with patch.object(mc, '_maybe_calibrate_size', return_value=[tf.constant(np.ndarray((3, 32, 32, 3))), tf.constant(np.ndarray((3, 32, 32, 3)))]) as mcs:
                with patch.object(mc, '_enas_cell', return_value='ec') as ec:
                    mc._enas_layer(0, [tf.constant(np.ndarray((3, 32, 32, 3))), tf.constant(np.ndarray((3, 32, 32, 3)))], [0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 2, 0, 3, 0, 3, 0, 4, 0, 4], 3)
                    mcs.assert_called()
                    ec.assert_called()
                    stack.assert_called()
                    gather.assert_called()
                    transpose.assert_called_with('gather', [1, 2, 3, 0, 4])
                    reshape.assert_called()
                    relu.assert_called_with('reshape')
                    conv2d.assert_called_with('relu', 'reshape', strides=[1, 1, 1, 1], padding='SAME', data_format="NHWC")
                    batch_norm.assert_called_with('conv2d', is_training=True, data_format='NHWC')
                    create_weight.assert_called_with('w', [7, 9])
                    to_int32.assert_called()

    @patch('src.cifar10.micro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.micro_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.micro_child.fw.conv2d', return_value="conv2d")
    @patch('src.cifar10.micro_child.fw.relu', return_value="relu")
    @patch('src.cifar10.micro_child.fw.reshape', return_value="reshape")
    @patch('src.cifar10.micro_child.fw.transpose', return_value="transpose")
    @patch('src.cifar10.micro_child.fw.gather', return_value="gather")
    @patch('src.cifar10.micro_child.fw.stack', return_value=tf.constant(np.ndarray((2, 3, 32, 32, 3))))
    @patch('src.cifar10.micro_child.fw.create_weight', return_value="fw.create_weight")
    def test_enas_layer_nchw(self, create_weight, stack, gather, transpose, reshape, relu, conv2d, batch_norm, to_int32):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            mc.data_format = "NCHW"
            with patch.object(mc, '_maybe_calibrate_size', return_value=[tf.constant(np.ndarray((3, 32, 32, 3))), tf.constant(np.ndarray((3, 32, 32, 3)))]) as mcs:
                with patch.object(mc, '_enas_cell', return_value='ec') as ec:
                    mc._enas_layer(0, [tf.constant(np.ndarray((3, 32, 32, 3))), tf.constant(np.ndarray((3, 32, 32, 3)))], [0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 2, 0, 3, 0, 3, 0, 4, 0, 4], 3)
                    mcs.assert_called()
                    ec.assert_called()
                    stack.assert_called()
                    gather.assert_called()
                    transpose.assert_called_with('gather', [1, 0, 2, 3, 4])
                    reshape.assert_called()
                    relu.assert_called_with('reshape')
                    conv2d.assert_called_with('relu', 'reshape', strides=[1, 1, 1, 1], padding='SAME', data_format="NCHW")
                    batch_norm.assert_called_with('conv2d', is_training=True, data_format='NCHW')
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
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
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
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
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
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
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
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
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

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.micro_child.fw.map_fn', return_value="map_fn")
    @patch('src.cifar10.micro_child.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.micro_child.fw.equal', return_value="equal")
    @patch('src.cifar10.micro_child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.micro_child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.micro_child.fw.shuffle_batch', return_value=(tf.constant(np.ndarray((2, 32, 32, 3))), "y_valid_shuffle"))
    @patch('src.cifar10.micro_child.print')
    def test_build_valid_rl(self, print, shuffle_batch, argmax, to_int32, equal, reduce_sum, map_fn, _Model):
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            mc.data_format = "NHWC"
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
        with patch('src.cifar10.micro_child.Model.__init__', new=mock_init):
            mc = MicroChild({}, {}, num_epochs=310, drop_path_keep_prob=0.9)
            with patch.object(mc, '_build_train', return_value='model') as build_train:
                with patch.object(mc, '_build_valid', return_value='model') as build_valid:
                    with patch.object(mc, '_build_test', return_value='model') as build_test:
                        mock_controller = mock.MagicMock()
                        mock_controller.sample_arc = (None, None)
                        mc.connect_controller(mock_controller)

if "__main__" == "__name__":
    unittest.main()
