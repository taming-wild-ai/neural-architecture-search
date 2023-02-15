import unittest
import unittest.mock as mock
from unittest.mock import patch

import numpy as np
from src.cifar10.general_child import tf
tf.compat.v1.disable_eager_execution()

from src.cifar10.general_child import GeneralChild


class TestGeneralChild(unittest.TestCase):
    @patch('src.cifar10.models.Model.__init__')
    def test_init(self, _Model):
        gc = GeneralChild({}, {})
        self.assertEqual(GeneralChild, type(gc))

    @patch('src.cifar10.models.Model.__init__')
    def test_get_hw(self, _Model):
        gc = GeneralChild({}, {})
        self.assertEqual(gc._get_HW(tf.constant(np.ndarray((1, 2, 3)))), 3)

    @patch('src.cifar10.models.Model.__init__')
    def test_get_strides_nhwc(self, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NHWC"
        self.assertEqual([1, 2, 2, 1], gc._get_strides(2))

    @patch('src.cifar10.models.Model.__init__')
    def test_get_strides_nchw(self, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NCHW"
        self.assertEqual([1, 1, 2, 2], gc._get_strides(2))

    @patch('src.cifar10.models.Model.__init__')
    def test_get_strides_exception(self, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "INVALID"
        self.assertRaises(ValueError, gc._get_strides, 2)

    @patch('src.cifar10.models.Model.__init__')
    def test_factorized_reduction_failure(self, _Model):
        gc = GeneralChild({}, {})
        self.assertRaises(AssertionError, gc._factorized_reduction, None, 3, None, None)

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.create_weight')
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value=None)
    @patch('src.cifar10.general_child.batch_norm')
    def test_factorized_reduction_stride1(self, batch_norm, conv2d, create_weight, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "DOESN'T MATTER"
        self.assertRaises(AssertionError, gc._factorized_reduction, None, 3, None, None)
        with patch.object(gc, '_get_C', return_value=None) as get_C:
            gc._factorized_reduction(None, 2, 1, True)
            get_C.assert_called_with(None)
            create_weight.assert_called_with("w", [1, 1, None, 2])
            batch_norm.assert_called_with(None, True, data_format="DOESN'T MATTER")

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.create_weight', return_value="w")
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.tf.nn.avg_pool', return_value="path1")
    @patch('src.cifar10.general_child.tf.pad', return_value=np.ndarray((5, 5, 5, 5)))
    @patch('src.cifar10.general_child.tf.concat', return_value="final_path")
    @patch('src.cifar10.general_child.batch_norm', return_value="final_path")
    def test_factorized_reduction_nhwc(self, batch_norm, concat, pad, avg_pool, conv2d, create_weight, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NHWC"
        with patch.object(gc, '_get_strides', return_value="stride spec") as get_strides:
            with patch.object(gc, '_get_C', return_value="inp_c") as get_C:
                self.assertEqual("final_path", gc._factorized_reduction(None, 2, 2, True))
                get_strides.assert_called_with(2)
                avg_pool.assert_called() # (None, ksize=[1, 1, 1, 1], strides="stride spec", padding="VALID", data_format="NHWC")
                get_C.assert_any_call("path1")
                create_weight.assert_called_with("w", [1, 1, "inp_c", 1])
                conv2d.assert_called() # ("path1", "w", [1, 1, 1, 1], "SAME", data_format="NHWC")
                pad.assert_called_with(None, [[0, 0], [0, 1], [0, 1], [0, 0]])
                avg_pool.assert_called() # (np.ndarray((5, 5, 5, 5))[:, 1:, 1:, :], [1, 1, 1, 1], "stride_spec", "VALID", data_format="NHWC")
                get_C.assert_any_call("path1")
                create_weight.assert_any_call("w", [1, 1, "inp_c", 1])
                conv2d.assert_called() # ("path1", "w", [1, 1, 1, 1], "SAME", data_format="NHWC")
                concat.assert_called_with(values=["conv2d", "conv2d"], axis=3)
                batch_norm.assert_any_call("final_path", True, data_format="NHWC")

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.create_weight', return_value="w")
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.tf.nn.avg_pool', return_value="path1")
    @patch('src.cifar10.general_child.tf.pad', return_value=np.ndarray((5, 5, 5, 5)))
    @patch('src.cifar10.general_child.tf.concat', return_value="final_path")
    @patch('src.cifar10.general_child.batch_norm', return_value="final_path")
    def test_factorized_reduction_nchw(self, batch_norm, concat, pad, avg_pool, conv2d, create_weight, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NCHW"
        with patch.object(gc, '_get_strides', return_value="stride spec") as get_strides:
            with patch.object(gc, '_get_C', return_value="inp_c") as get_C:
                self.assertEqual("final_path", gc._factorized_reduction(None, 2, 2, True))
                get_strides.assert_called_with(2)
                avg_pool.assert_called() # (None, ksize=[1, 1, 1, 1], strides="stride spec", padding="VALID", data_format="NHWC")
                get_C.assert_any_call("path1")
                create_weight.assert_called_with("w", [1, 1, "inp_c", 1])
                conv2d.assert_called() # ("path1", "w", [1, 1, 1, 1], "SAME", data_format="NHWC")
                pad.assert_called_with(None, [[0, 0], [0, 0], [0, 1], [0, 1]])
                avg_pool.assert_called() # (np.ndarray((5, 5, 5, 5))[:, 1:, 1:, :], [1, 1, 1, 1], "stride_spec", "VALID", data_format="NHWC")
                get_C.assert_any_call("path1")
                create_weight.assert_any_call("w", [1, 1, "inp_c", 1])
                conv2d.assert_called() # ("path1", "w", [1, 1, 1, 1], "SAME", data_format="NHWC")
                concat.assert_called_with(values=["conv2d", "conv2d"], axis=1)
                batch_norm.assert_any_call("final_path", True, data_format="NCHW")

    @patch('src.cifar10.models.Model.__init__')
    def test_get_c_nchw(self, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NCHW"
        self.assertEqual(3, gc._get_C(tf.constant(np.ndarray((45000, 3, 32, 32)))))

    @patch('src.cifar10.models.Model.__init__')
    def test_get_c_nhwc(self, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NHWC"
        self.assertEqual(3, gc._get_C(tf.constant(np.ndarray((45000, 32, 32, 3)))))

    @patch('src.cifar10.models.Model.__init__')
    def test_get_c_raises(self, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "INVALID"
        self.assertRaises(ValueError, gc._get_C, tf.constant(np.ndarray((4, 32, 32, 3))))

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.print')
    @patch('src.cifar10.general_child.create_weight', return_value="w")
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.batch_norm', return_value="final_path")
    @patch('src.cifar10.general_child.global_avg_pool', return_value="global_avg_pool")
    @patch('src.cifar10.general_child.tf.nn.dropout')
    def test_model_nhwc_KNOWN_TO_FAIL(self, dropout, global_avg_pool, batch_norm, conv2d, create_weight, _print, _Model):
        gc = GeneralChild({}, {})
        gc.name = "generic_model"
        gc.data_format = "NHWC"
        gc.keep_prob = 0.9
        with patch.object(gc, '_enas_layer') as enas_layer:
            gc._model({}, True)
            create_weight.assert_called_with("w", [3, 3, 3, 4])
            conv2d.assert_called()
            enas_layer.assert_called_with(0, 0, 0, 0, True)
            global_avg_pool.assert_called_with(None, data_format="NHWC")
            dropout.assert_called_with(None, 0.9)

    @patch('src.cifar10.general_child.print')
    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.create_weight', return_value="w")
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.batch_norm', return_value="final_path")
    @patch('src.cifar10.general_child.global_avg_pool', return_value="global_avg_pool")
    @patch('src.cifar10.general_child.tf.nn.dropout', return_value=tf.constant(np.ndarray((4, 3, 32, 32))))
    @patch('src.cifar10.general_child.tf.matmul')
    def test_model_nchw_no_whole_channels(self, matmul, dropout, global_avg_pool, batch_norm, conv2d, create_weight, _Model, _print):
        gc = GeneralChild({}, {})
        gc.name = "generic_model"
        gc.data_format = "NCHW"
        gc.keep_prob = 0.9
        gc.pool_layers = [0]
        with patch.object(gc, '_enas_layer', return_value="enas_layer") as enas_layer:
            with patch.object(gc, '_factorized_reduction', return_value="factorized_reduction") as factorized_reduction:
                gc._model({}, True)
                create_weight.assert_any_call("w", [3, 3, 3, 24])
                conv2d.assert_called()
                batch_norm.assert_called_with("conv2d", True, data_format="NCHW")
                enas_layer.assert_called_with(1, ['factorized_reduction', 'factorized_reduction', 'enas_layer'], 18, 24, True)
                factorized_reduction.assert_called_with('enas_layer', 24, 2, True)
                global_avg_pool.assert_called_with('enas_layer', data_format="NCHW")
                dropout.assert_called_with('global_avg_pool', 0.9)
                create_weight.assert_called_with("w", [3, 10])
                matmul.assert_called()

    @patch('src.cifar10.general_child.print')
    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.create_weight', return_value="w")
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.batch_norm', return_value="final_path")
    @patch('src.cifar10.general_child.global_avg_pool', return_value="global_avg_pool")
    @patch('src.cifar10.general_child.tf.nn.dropout', return_value=tf.constant(np.ndarray((4, 3, 32, 32))))
    @patch('src.cifar10.general_child.tf.matmul')
    def test_model_nchw_whole_channels(self, matmul, dropout, global_avg_pool, batch_norm, conv2d, create_weight, _Model, _print):
        gc = GeneralChild({}, {})
        gc.name = "generic_model"
        gc.data_format = "NCHW"
        gc.keep_prob = 0.9
        gc.pool_layers = [0]
        gc.whole_channels = True
        gc.fixed_arc = ""
        with patch.object(gc, '_fixed_layer', return_value="enas_layer") as fixed_layer:
            with patch.object(gc, '_factorized_reduction', return_value="factorized_reduction") as factorized_reduction:
                gc._model({}, True)
                create_weight.assert_any_call("w", [3, 3, 3, 24])
                conv2d.assert_called()
                batch_norm.assert_called_with("conv2d", True, data_format="NCHW")
                fixed_layer.assert_called_with(1, ['factorized_reduction', 'factorized_reduction', 'enas_layer'], 1, 48, True)
                factorized_reduction.assert_called_with('enas_layer', 48, 2, True)
                global_avg_pool.assert_called_with('enas_layer', data_format="NCHW")
                dropout.assert_called_with('global_avg_pool', 0.9)
                create_weight.assert_called_with("w", [3, 10])
                matmul.assert_called()

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.tf.case', return_value=tf.constant(np.ndarray((4, 32, 32, 24)), tf.float32))
    @patch('src.cifar10.general_child.tf.add_n', return_value="add_n")
    @patch('src.cifar10.general_child.batch_norm', return_value="batch_norm")
    def test_enas_layer_whole_channels_nhwc(self, batch_norm, add_n, case, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NHWC"
        gc.whole_channels = True
        gc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0".split(" ") if x])
        with patch.object(gc, '_conv_branch', return_value="conv_branch") as conv_branch:
            with patch.object(gc, '_pool_branch', return_value="pool_branch") as pool_branch:
                input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
                gc._enas_layer(1, [input_tensor], 0, 24, True)
                conv_branch.assert_called_with(input_tensor, 5, True, 24, 24, start_idx=0, separable=True)
                case.assert_called()
                batch_norm.assert_called_with("add_n", True, data_format="NHWC")

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.tf.case', return_value=tf.constant(np.ndarray((4, 24, 32, 32)), tf.float32))
    @patch('src.cifar10.general_child.tf.add_n', return_value="add_n")
    @patch('src.cifar10.general_child.batch_norm', return_value="batch_norm")
    def test_enas_layer_whole_channels_nchw(self, batch_norm, add_n, case, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NCHW"
        gc.whole_channels = True
        gc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0".split(" ") if x])
        with patch.object(gc, '_conv_branch', return_value="conv_branch") as conv_branch:
            with patch.object(gc, '_pool_branch', return_value="pool_branch") as pool_branch:
                input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
                gc._enas_layer(1, [input_tensor], 0, 24, True)
                conv_branch.assert_called_with(input_tensor, 5, True, 24, 24, start_idx=0, separable=True)
                case.assert_called()
                batch_norm.assert_called_with("add_n", True, data_format="NCHW")

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.tf.concat', return_value="concat")
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.tf.nn.relu', return_value="relu")
    @patch('src.cifar10.general_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.general_child.tf.add_n', return_value="add_n")
    def test_enas_layer_not_whole_channels_nhwc(self, add_n, batch_norm, relu, conv_2d, concat, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NHWC"
        gc.whole_channels = False
        gc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0 2 0 0 1 2 0 0 0 0".split(" ") if x])
        with patch.object(gc, '_conv_branch', return_value="conv_branch") as conv_branch:
            with patch.object(gc, '_pool_branch', return_value="pool_branch") as pool_branch:
                input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
                gc._enas_layer(1, [input_tensor], 0, 24, True)
                conv_branch.assert_called_with(input_tensor, 5, True, 0, 24, start_idx=2, separable=True)
                concat.assert_called()
                conv_2d.assert_called()
                relu.assert_called_with("batch_norm")
                batch_norm.assert_called_with("add_n", True, data_format="NHWC")

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.tf.concat', return_value="concat")
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.tf.nn.relu', return_value="relu")
    @patch('src.cifar10.general_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.general_child.tf.add_n', return_value="add_n")
    @patch('src.cifar10.general_child.create_weight', return_value="create_weight")
    @patch('src.cifar10.general_child.tf.logical_and', return_value="logical_and")
    @patch('src.cifar10.general_child.tf.logical_or', return_value="logical_or")
    @patch('src.cifar10.general_child.tf.boolean_mask', return_value="boolean_mask")
    @patch('src.cifar10.general_child.tf.reshape', return_value="reshape")
    @patch('src.cifar10.general_child.tf.shape', return_value=["shape"])
    def test_enas_layer_not_whole_channels_nchw(self, shape, reshape, boolean_mask, logical_or, logical_and, create_weight, add_n, batch_norm, relu, conv_2d, concat, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NCHW"
        gc.whole_channels = False
        gc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0 2 0 0 1 2 0 0 0 0".split(" ") if x])
        with patch.object(gc, '_conv_branch', return_value="conv_branch") as conv_branch:
            with patch.object(gc, '_pool_branch', return_value="pool_branch") as pool_branch:
                input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
                gc._enas_layer(1, [input_tensor], 0, 24, True)
                conv_branch.assert_called_with(input_tensor, 5, True, 0, 24, start_idx=2, separable=True)
                concat.assert_called()
                conv_2d.assert_called()
                relu.assert_called_with("batch_norm")
                batch_norm.assert_called_with("add_n", True, data_format="NCHW")
                create_weight.assert_called_with("w", [144, 24])
                logical_and.assert_called()
                logical_or.assert_called_with("logical_or", "logical_and")
                boolean_mask.assert_called_with("create_weight", "logical_or")
                shape.assert_called()
                reshape.assert_called_with("concat", ['shape', -1, 32, 3])

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.tf.nn.relu', return_value="relu")
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.batch_norm', return_value="batch_norm")
    def test_fixed_layer_whole_channels_nhwc(self, batch_norm, conv2d, relu, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NHWC"
        gc.whole_channels = True
        gc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0".split(" ") if x])
        input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
        gc._fixed_layer(0, [input_tensor], 0, 24, True)
        relu.assert_called_with("batch_norm")
        conv2d.assert_called()
        batch_norm.assert_called_with("conv2d", True, data_format="NHWC")

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.tf.nn.relu', return_value="relu")
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.general_child.create_weight', return_value="create_weight")
    @patch('src.cifar10.general_child.tf.concat', return_value="concat")
    def test_fixed_layer_whole_channels_nhwc_second_layer(self, concat, create_weight, batch_norm, conv2d, relu, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NHWC"
        gc.whole_channels = True
        gc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0".split(" ") if x])
        input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
        gc._fixed_layer(1, [input_tensor], 0, 24, True)
        create_weight.assert_called_with("w", [1, 1, 96, 24])
        relu.assert_called_with("concat")
        conv2d.assert_called()
        batch_norm.assert_called_with("conv2d", True, data_format="NHWC")

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.tf.nn.relu', return_value="relu")
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.general_child.create_weight', return_value="create_weight")
    def test_fixed_layer_whole_channels_nchw(self, create_weight, batch_norm, conv2d, relu, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NCHW"
        gc.whole_channels = True
        gc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0".split(" ") if x])
        input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
        gc._fixed_layer(0, [input_tensor], 0, 24, True)
        relu.assert_called_with("batch_norm")
        conv2d.assert_called()
        batch_norm.assert_called_with("conv2d", True, data_format="NCHW")

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.tf.nn.relu', return_value="relu")
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.general_child.create_weight', return_value="create_weight")
    def test_fixed_layer_whole_channels_nchw_raises(self, create_weight, batch_norm, conv2d, relu, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NCHW"
        gc.whole_channels = True
        gc.sample_arc = np.array([int(x) for x in "6 3 0 0 1 0".split(" ") if x])
        input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
        self.assertRaises(ValueError, gc._fixed_layer, 0, [input_tensor], 0, 24, True)

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.create_weight')
    @patch('src.cifar10.general_child.tf.nn.relu', return_value="relu")
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.general_child.tf.concat', return_value="concat")
    def test_fixed_layer_not_whole_channels_nhwc(self, concat, batch_norm, conv2d, relu, _create_weight, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NHWC"
        gc.whole_channels = False
        gc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0 2 0 0 1 2 0 0 0 0".split(" ") if x])
        input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
        with patch.object(gc, '_conv_branch', return_value="conv_branch") as conv_branch:
            with patch.object(gc, '_pool_branch', return_value="pool_branch") as pool_branch:
                gc._fixed_layer(0, [input_tensor], 0, 24, True)
                conv_branch.assert_called_with(input_tensor, 5, True, 0, separable=True)
                relu.assert_called_with("concat")
                conv2d.assert_called()
                batch_norm.assert_called_with("conv2d", True, data_format="NHWC")
                pool_branch.assert_called_with(input_tensor, True, 0, "max")
                concat.assert_called_with(['conv_branch', 'conv_branch', 'conv_branch', 'conv_branch', 'pool_branch', 'pool_branch'], axis=3)

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.create_weight')
    @patch('src.cifar10.general_child.tf.nn.relu', return_value="relu")
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.general_child.tf.concat', return_value="concat")
    def test_fixed_layer_not_whole_channels_nchw_second_layer(self, concat, batch_norm, conv2d, relu, _create_weight, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NCHW"
        gc.whole_channels = False
        gc.sample_arc = np.array([int(x) for x in "0 3 0 0 1 0 2 0 0 1 2 0 0 0 0".split(" ") if x])
        input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
        with patch.object(gc, '_conv_branch', return_value="conv_branch") as conv_branch:
            with patch.object(gc, '_pool_branch', return_value="pool_branch") as pool_branch:
                gc._fixed_layer(1, [input_tensor], 0, 24, True)
                conv_branch.assert_called_with(input_tensor, 5, True, 0, separable=True)
                relu.assert_called_with("concat")
                conv2d.assert_called()
                batch_norm.assert_called_with("conv2d", True, data_format="NCHW")
                pool_branch.assert_called_with(input_tensor, True, 0, "max")
                concat.assert_called_with(['batch_norm'], axis=1)

    @patch('src.cifar10.models.Model.__init__')
    def test_conv_branch_failure(self, _Model):
        gc = GeneralChild({}, {})
        self.assertRaises(AssertionError, gc._conv_branch, None, 24, True, 0, 24)

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.general_child.tf.nn.relu', return_value="relu")
    def test_conv_branch_nhwc(self, relu, batch_norm, conv2d, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NHWC"
        gc.fixed_arc = "0 3 0 0 1 0"
        input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
        gc._conv_branch(input_tensor, 24, True, 0, 24)
        conv2d.assert_called()
        batch_norm.assert_called_with("conv2d", True, data_format="NHWC")
        relu.assert_called_with("batch_norm")

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.create_weight', return_value="create_weight")
    @patch('src.cifar10.general_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.general_child.tf.nn.relu', return_value="relu")
    def test_conv_branch_nchw(self, relu, batch_norm, create_weight, conv2d, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NCHW"
        gc.fixed_arc = "0 3 0 0 1 0"
        input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
        gc._conv_branch(input_tensor, 24, True, 0, 24)
        create_weight.assert_any_call("w", [1, 1, 3, 24])
        conv2d.assert_called()
        batch_norm.assert_called_with("conv2d", True, data_format="NCHW")
        relu.assert_called_with("batch_norm")

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.create_weight', return_value=tf.Variable(initial_value=np.zeros((24, 24, 24, 1))))
    @patch('src.cifar10.general_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.general_child.tf.nn.relu', return_value="relu")
    @patch('src.cifar10.general_child.tf.transpose', return_value="transpose")
    @patch('src.cifar10.general_child.tf.reshape', return_value="reshape")
    @patch('src.cifar10.general_child.tf.nn.separable_conv2d', return_value="sep_conv2d")
    @patch('src.cifar10.general_child.batch_norm_with_mask', return_value="bnwm")
    def test_conv_branch_nchw_separable(self, bnwm, sep_conv2d, reshape, transpose, relu, batch_norm, create_weight, conv2d, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NCHW"
        gc.fixed_arc = "0 3 0 0 1 0"
        gc.filter_size = 24
        input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
        gc._conv_branch(input_tensor, 24, True, 0, 24, ch_mul=1, start_idx=None, separable=True)
        create_weight.assert_any_call("w", [1, 1, 3, 24])
        conv2d.assert_called()
        batch_norm.assert_called_with("sep_conv2d", True, data_format="NCHW")
        relu.assert_called_with("batch_norm")

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.create_weight', return_value=tf.Variable(initial_value=np.zeros((24, 24, 24, 1))))
    @patch('src.cifar10.general_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.general_child.tf.nn.relu', return_value="relu")
    @patch('src.cifar10.general_child.tf.transpose', return_value=tf.Variable(initial_value=np.zeros((24, 24, 24, 1))))
    @patch('src.cifar10.general_child.tf.reshape', return_value="reshape")
    @patch('src.cifar10.general_child.tf.nn.separable_conv2d', return_value="sep_conv2d")
    @patch('src.cifar10.general_child.batch_norm_with_mask', return_value="bnwm")
    def test_conv_branch_nchw_second_index(self, bnwm, sep_conv2d, reshape, transpose, relu, batch_norm, create_weight, conv2d, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NCHW"
        gc.fixed_arc = "0 3 0 0 1 0"
        input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
        gc._conv_branch(input_tensor, 24, True, 0, 24, ch_mul=1, start_idx=1)
        create_weight.assert_any_call("w", [1, 1, 3, 24])
        conv2d.assert_called()
        batch_norm.assert_called_with("conv2d", True, data_format="NCHW")
        relu.assert_called_with("bnwm")

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.create_weight', return_value=tf.Variable(initial_value=np.zeros((24, 24, 24, 1))))
    @patch('src.cifar10.general_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.general_child.tf.nn.relu', return_value="relu")
    @patch('src.cifar10.general_child.tf.transpose', return_value="transpose")
    @patch('src.cifar10.general_child.tf.reshape', return_value="reshape")
    @patch('src.cifar10.general_child.tf.nn.separable_conv2d', return_value="sep_conv2d")
    @patch('src.cifar10.general_child.batch_norm_with_mask', return_value="bnwm")
    def test_conv_branch_nchw_second_index_separable(self, bnwm, sep_conv2d, reshape, transpose, relu, batch_norm, create_weight, conv2d, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NCHW"
        gc.fixed_arc = "0 3 0 0 1 0"
        input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
        gc._conv_branch(input_tensor, 24, True, 0, 24, ch_mul=1, start_idx=1, separable=True)
        create_weight.assert_any_call("w", [1, 1, 3, 24])
        conv2d.assert_called()
        batch_norm.assert_called_with("conv2d", True, data_format="NCHW")
        relu.assert_called_with("bnwm")

    @patch('src.cifar10.models.Model.__init__')
    def test_pool_branch_failure(self, _Model):
        gc = GeneralChild({}, {})
        self.assertRaises(AssertionError, gc._pool_branch, None, None, None, None)

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.general_child.tf.nn.relu', return_value="relu")
    @patch('src.cifar10.general_child.tf.compat.v1.layers')
    @patch('src.cifar10.general_child.create_weight', return_value=tf.Variable(initial_value=np.zeros((24, 24, 24, 1))))
    def test_pool_branch_nhwc_avg(self, create_weight, layers, relu, batch_norm, conv2d, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NHWC"
        gc.fixed_arc = "0 3 0 0 1 0"
        input_tensor = tf.constant(np.ndarray((4, 32, 32, 3)), name="hashable")
        with patch.object(layers, 'average_pooling2d'):
            gc._pool_branch(input_tensor, True, 0, "avg", start_idx=True)
            conv2d.assert_called()
            batch_norm.assert_called_with("conv2d", True, data_format="NHWC")
            relu.assert_called_with("batch_norm")
            layers.average_pooling2d.assert_called_with("relu", [3, 3], [1, 1], "SAME", data_format="channels_last")

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.tf.nn.conv2d', return_value="conv2d")
    @patch('src.cifar10.general_child.batch_norm', return_value="batch_norm")
    @patch('src.cifar10.general_child.tf.nn.relu', return_value="relu")
    @patch('src.cifar10.general_child.tf.compat.v1.layers')
    @patch('src.cifar10.general_child.create_weight', return_value=tf.Variable(initial_value=np.zeros((24, 24, 24, 1))))
    def test_pool_branch_nchw_max(self, create_weight, layers, relu, batch_norm, conv2d, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NCHW"
        gc.fixed_arc = "0 3 0 0 1 0"
        input_tensor = tf.constant(np.ndarray((4, 3, 32, 32)), name="hashable")
        with patch.object(layers, 'average_pooling2d'):
            gc._pool_branch(input_tensor, True, 0, "max", start_idx=True)
            conv2d.assert_called()
            batch_norm.assert_called_with("conv2d", True, data_format="NCHW")
            relu.assert_called_with("batch_norm")
            layers.average_pooling2d.assert_not_called()

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.tf.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.general_child.tf.equal', return_value="equal")
    @patch('src.cifar10.general_child.tf.compat.v1.to_int32', return_value="to_int32")
    @patch('src.cifar10.general_child.print')
    @patch('src.cifar10.general_child.tf.nn.sparse_softmax_cross_entropy_with_logits', return_value="sscewl")
    @patch('src.cifar10.general_child.tf.reduce_mean', return_value="reduce_mean")
    @patch('src.cifar10.general_child.tf.argmax', return_value=10)
    @patch('src.cifar10.general_child.get_train_ops', return_value=(1, 2, 3, 4))
    def test_build_train(self, get_train_ops, argmax, reduce_mean, sscewl, print, to_int32, equal, reduce_sum, _Model):
        gc = GeneralChild({}, {})
        gc.x_train = 1
        gc.y_train = 2
        gc.clip_mode = None
        gc.grad_bound = None
        gc.l2_reg = 1e-4
        gc.lr_init = 0.1
        gc.lr_dec_start = 0
        gc.lr_dec_every=100
        gc.lr_dec_rate = 0.1
        gc.num_train_batches = 310
        gc.optim_algo = None
        gc.sync_replicas = False
        gc.num_aggregate = None
        gc.num_replicas = None
        gc.name = "general_child"
        with patch.object(gc, '_model', return_value="model") as model:
            gc._build_train()
            print.assert_any_call("-" * 80)
            print.assert_any_call("Build train graph")
            model.assert_called_with(gc.x_train, is_training=True)
            sscewl.assert_called_with(logits="model", labels=gc.y_train)
            reduce_mean.assert_called_with("sscewl")
            argmax.assert_called_with("model", axis=1)
            get_train_ops.assert_called()
            to_int32.assert_called_with('equal')
            equal.assert_called_with('to_int32', 2)

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.print')
    @patch('src.cifar10.general_child.tf.argmax', return_value="argmax")
    @patch('src.cifar10.general_child.tf.compat.v1.to_int32', return_value="to_int32")
    @patch('src.cifar10.general_child.tf.equal', return_value="equal")
    @patch('src.cifar10.general_child.tf.reduce_sum', return_value="reduce_sum")
    def test_build_valid(self, reduce_sum, equal, to_int32, argmax, print, _Model):
        gc = GeneralChild({}, {})
        gc.x_valid = True
        gc.y_valid = None
        with patch.object(gc, '_model', return_value='model') as model:
            gc._build_valid()
            print.assert_any_call("-" * 80)
            print.assert_any_call("Build valid graph")
            model.assert_called_with(True, False, reuse=True)
            argmax.assert_called_with('model', axis=1)
            to_int32.assert_any_call('argmax')
            equal.assert_called_with('to_int32', None)
            to_int32.assert_any_call('equal')
            reduce_sum.assert_called_with('to_int32')

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.print')
    @patch('src.cifar10.general_child.tf.argmax', return_value="argmax")
    @patch('src.cifar10.general_child.tf.compat.v1.to_int32', return_value="to_int32")
    @patch('src.cifar10.general_child.tf.equal', return_value="equal")
    @patch('src.cifar10.general_child.tf.reduce_sum', return_value="reduce_sum")
    def test_build_test(self, reduce_sum, equal, to_int32, argmax, print, _Model):
        gc = GeneralChild({}, {})
        gc.x_test = True
        gc.y_test = False
        with patch.object(gc, '_model', return_value='model') as model:
            gc._build_test()
            print.assert_any_call('-' * 80)
            print.assert_any_call("Build test graph")
            model.assert_called_with(True, False, reuse=True)
            argmax.assert_called_with('model', axis=1)
            to_int32.assert_any_call('argmax')
            equal.assert_called_with('to_int32', False)
            to_int32.assert_any_call('equal')
            reduce_sum.assert_called_with('to_int32')

    @patch('src.cifar10.models.Model.__init__')
    @patch('src.cifar10.general_child.tf.map_fn', return_value="map_fn")
    @patch('src.cifar10.general_child.tf.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.general_child.tf.equal', return_value="equal")
    @patch('src.cifar10.general_child.tf.compat.v1.to_int32', return_value="to_int32")
    @patch('src.cifar10.general_child.tf.argmax', return_value="argmax")
    @patch('src.cifar10.general_child.tf.compat.v1.train.shuffle_batch', return_value=(tf.constant(np.ndarray((2, 3, 32, 32))), "y_valid_shuffle"))
    @patch('src.cifar10.general_child.print')
    def test_build_valid_rl(self, print, shuffle_batch, argmax, to_int32, equal, reduce_sum, map_fn, _Model):
        gc = GeneralChild({}, {})
        gc.data_format = "NCHW"
        gc.images = { 'valid_original': np.ndarray((1, 3, 32, 32)) }
        gc.labels = { 'valid_original': np.ndarray((1)) }
        gc.batch_size = 32
        gc.seed = None
        with patch.object(gc, '_model', return_value='model') as model:
            gc.build_valid_rl(shuffle=True)
            print.assert_any_call('-' * 80)
            print.assert_any_call('Build valid graph on shuffled data')
            shuffle_batch.assert_called_with(
                [gc.images['valid_original'], gc.labels['valid_original']],
                batch_size=gc.batch_size,
                capacity=25000,
                enqueue_many=True,
                min_after_dequeue=0,
                num_threads=16,
                seed=gc.seed,
                allow_smaller_final_batch=True)
            model.assert_called()
            argmax.assert_called_with("model", axis=1)
            to_int32.assert_any_call("argmax")
            equal.assert_called_with("to_int32", "y_valid_shuffle")
            to_int32.assert_any_call("equal")
            reduce_sum.assert_called_with("to_int32")
            map_fn.assert_called()

    @patch('src.cifar10.models.Model.__init__')
    def test_connect_controller_no_fixed_arc(self, _Model):
        gc = GeneralChild({}, {})
        with patch.object(gc, '_build_train') as build_train:
            with patch.object(gc, '_build_valid') as build_valid:
                with patch.object(gc, '_build_test') as build_test:
                    controller_mock = mock.MagicMock()
                    gc.connect_controller(controller_mock)
                    build_train.assert_called_with()
                    build_valid.assert_called_with()
                    build_test.assert_called_with()

    @patch('src.cifar10.models.Model.__init__')
    def test_connect_controller_fixed_arc(self, _Model):
        gc = GeneralChild({}, {})
        gc.fixed_arc = ""
        with patch.object(gc, '_build_train') as build_train:
            with patch.object(gc, '_build_valid') as build_valid:
                with patch.object(gc, '_build_test') as build_test:
                    controller_mock = mock.MagicMock()
                    gc.connect_controller(controller_mock)
                    build_train.assert_called_with()
                    build_valid.assert_called_with()
                    build_test.assert_called_with()
    # @patch('src.cifar10.models.print')
    # @patch('src.cifar10.general_child.tf.compat.v1.train.shuffle_batch')
    # @patch('src.cifar10.main.tf.map_fn')
    # @patch('src.cifar10.general_child.tf.compat.v1.train.batch')
    # def test_nhwc(self, batch, map_fn, shuffle_batch, print):
    #     mock_images = {
    #         "train": [np.zeros((32, 32, 3))],
    #         "valid": None,
    #         "test": np.zeros((1, 1, 1))
    #     }
    #     mock_labels = {
    #         "train": [np.zeros((32, 32, 3))],
    #         "test": np.zeros((1, 1, 1))
    #     }
    #     shuffle_batch.return_value = (mock_images['train'], mock_labels['train'])
    #     map_fn.return_value = mock_images['train']
    #     batch.return_value = (mock_images['train'], mock_labels['train'])
    #     gc = GeneralChild(mock_images, mock_labels)
    #     self.assertEqual(GeneralChild, type(gc))
    #     self.assertEqual(3, gc._get_C(tf.constant(np.zeros((45000, 32, 32, 3)))))
    #     self.assertEqual(32, gc._get_HW(tf.constant(np.zeros((45000, 32, 32, 3)))))
    #     self.assertEqual([1, 2, 2, 1], gc._get_strides(2))
    #     print.assert_any_call("-" * 80)
    #     print.assert_any_call("Build model child")
    #     print.assert_any_call("Build data ops")

    # @patch('src.cifar10.models.print')
    # @patch('src.cifar10.general_child.tf.compat.v1.train.shuffle_batch')
    # @patch('src.cifar10.main.tf.map_fn')
    # @patch('src.cifar10.general_child.tf.compat.v1.train.batch')
    # def test_nchw(self, batch, map_fn, shuffle_batch, print):
    #     mock_images = {
    #         "train": [np.zeros((45000, 3, 32, 32))],
    #         "valid": None,
    #         "test": np.zeros((45000, 1, 1, 1))
    #     }
    #     mock_labels = {
    #         "train": [np.zeros((45000, 3, 32, 32))],
    #         "test": np.zeros((45000, 1, 1, 1))
    #     }
    #     shuffle_batch.return_value = (mock_images['train'], mock_labels['train'])
    #     map_fn.return_value = mock_images['train']
    #     batch.return_value = (mock_images['train'], mock_labels['train'])
    #     gc = GeneralChild(mock_images, mock_labels, data_format="NCHW")
    #     self.assertEqual(GeneralChild, type(gc))
    #     self.assertEqual(3, gc._get_C(tf.constant(np.zeros((45000, 3, 32, 32)))))
    #     self.assertEqual(32, gc._get_HW(tf.constant(np.zeros((45000, 3, 32, 32)))))
    #     self.assertEqual([1, 1, 2, 2], gc._get_strides(2))
    #     print.assert_any_call("-" * 80)
    #     print.assert_any_call("Build model child")
    #     print.assert_any_call("Build data ops")

    # @patch('src.cifar10.models.print')
    # @patch('src.cifar10.general_child.tf.compat.v1.train.shuffle_batch')
    # @patch('src.cifar10.main.tf.map_fn')
    # @patch('src.cifar10.general_child.tf.compat.v1.train.batch')
    # def test_invalid_format(self, batch, map_fn, shuffle_batch, print):
    #     mock_images = {
    #         "train": [np.zeros((45000, 3, 32, 32))],
    #         "valid": None,
    #         "test": np.zeros((45000, 1, 1, 1))
    #     }
    #     mock_labels = {
    #         "train": [np.zeros((45000, 3, 32, 32))],
    #         "test": np.zeros((45000, 1, 1, 1))
    #     }
    #     shuffle_batch.return_value = (mock_images['train'], mock_labels['train'])
    #     map_fn.return_value = mock_images['train']
    #     batch.return_value = (mock_images['train'], mock_labels['train'])
    #     gc = GeneralChild(mock_images, mock_labels, data_format="INVALID")
    #     self.assertRaises(ValueError, gc._get_strides, 2)
    #     self.assertRaises(ValueError, gc._get_C, tf.constant(np.zeros((45000, 32, 32, 3))))

    # @patch('src.cifar10.models.print')
    # @patch('src.cifar10.general_child.tf.compat.v1.train.shuffle_batch')
    # @patch('src.cifar10.main.tf.map_fn')
    # @patch('src.cifar10.general_child.tf.compat.v1.train.batch')
    # def test_factorized_reduction_stride1(self, batch, map_fn, shuffle_batch, print):
    #     mock_images = {
    #         "train": [np.zeros((45000, 3, 32, 32))],
    #         "valid": None,
    #         "test": np.zeros((45000, 1, 1, 1))
    #     }
    #     mock_labels = {
    #         "train": [np.zeros((45000, 3, 32, 32))],
    #         "test": np.zeros((45000, 1, 1, 1))
    #     }
    #     shuffle_batch.return_value = (mock_images['train'], mock_labels['train'])
    #     map_fn.return_value = mock_images['train']
    #     batch.return_value = (mock_images['train'], mock_labels['train'])
    #     gc = GeneralChild(mock_images, mock_labels)
    #     gc._factorized_reduction(tf.constant(np.zeros((45000, 32, 32, 3)), tf.float32), 2, 1, True)

    # @patch('src.cifar10.models.print')
    # @patch('src.cifar10.general_child.tf.compat.v1.train.shuffle_batch')
    # @patch('src.cifar10.main.tf.map_fn')
    # @patch('src.cifar10.general_child.tf.compat.v1.train.batch')
    # def test_factorized_reduction_stride2(self, batch, map_fn, shuffle_batch, print):
    #     mock_images = {
    #         "train": [np.zeros((45000, 3, 32, 32))],
    #         "valid": None,
    #         "test": np.zeros((45000, 1, 1, 1))
    #     }
    #     mock_labels = {
    #         "train": [np.zeros((45000, 3, 32, 32))],
    #         "test": np.zeros((45000, 1, 1, 1))
    #     }
    #     shuffle_batch.return_value = (mock_images['train'], mock_labels['train'])
    #     map_fn.return_value = mock_images['train']
    #     batch.return_value = (mock_images['train'], mock_labels['train'])
    #     gc = GeneralChild(mock_images, mock_labels)
    #     gc._factorized_reduction(tf.constant(np.zeros((45000, 32, 32, 3)), tf.float32), 2, 2, True)

    # @patch('src.cifar10.models.print')
    # @patch('src.cifar10.general_child.print')
    # @patch('src.cifar10.general_child.tf.compat.v1.train.shuffle_batch')
    # @patch('src.cifar10.main.tf.map_fn')
    # @patch('src.cifar10.general_child.tf.compat.v1.train.batch')
    # def test_connect_controller_with_fixed_arc_KNOWN_TO_FAIL1(self, batch, map_fn, shuffle_batch, _print1, _print2):
    #     mock_images = {
    #         "train": np.zeros((45000, 32, 32, 3)),
    #         "valid": None,
    #         "test": np.zeros((45000, 1, 1, 1))
    #     }
    #     mock_labels = {
    #         "train": np.zeros((45000, 32, 32, 3)),
    #         "test": np.zeros((45000, 1, 1, 1))
    #     }
    #     shuffle_batch.return_value = (mock_images['train'], mock_labels['train'])
    #     map_fn.return_value = mock_images['train']
    #     batch.return_value = (mock_images['train'], mock_labels['train'])
    #     fixed_arc = "0 3 0 0 1 0"
    #     gc = GeneralChild(mock_images, mock_labels, whole_channels=True, fixed_arc=fixed_arc)
    #     controller = mock.MagicMock()
    #     controller.sample_arc = np.array([int(x) for x in fixed_arc.split(" ") if x])
    #     gc.connect_controller(controller)

    # @patch('src.cifar10.models.print')
    # @patch('src.cifar10.general_child.print')
    # @patch('src.cifar10.general_child.tf.compat.v1.train.shuffle_batch')
    # @patch('src.cifar10.main.tf.map_fn')
    # @patch('src.cifar10.general_child.tf.compat.v1.train.batch')
    # def test_connect_controller_with_fixed_arc_KNOWN_TO_FAIL2(self, batch, map_fn, shuffle_batch, _print1, _print2):
    #     mock_images = {
    #         "train": np.zeros((45000, 32, 32, 3)),
    #         "valid": None,
    #         "test": np.zeros((45000, 1, 1, 1))
    #     }
    #     mock_labels = {
    #         "train": np.zeros((45000, 32, 32, 3)),
    #         "test": np.zeros((45000, 1, 1, 1))
    #     }
    #     shuffle_batch.return_value = (mock_images['train'], mock_labels['train'])
    #     map_fn.return_value = mock_images['train']
    #     batch.return_value = (mock_images['train'], mock_labels['train'])
    #     fixed_arc = "0 3 0 0 1 0"
    #     gc = GeneralChild(mock_images, mock_labels, whole_channels=True, fixed_arc=fixed_arc, keep_prob=0.9)
    #     controller = mock.MagicMock()
    #     controller.sample_arc = np.array([int(x) for x in fixed_arc.split(" ") if x])
    #     gc.connect_controller(controller)

    # @patch('src.cifar10.models.print')
    # @patch('src.cifar10.general_child.print')
    # @patch('src.cifar10.general_child.tf.compat.v1.train.shuffle_batch')
    # @patch('src.cifar10.main.tf.map_fn')
    # @patch('src.cifar10.general_child.tf.compat.v1.train.batch')
    # def test_connect_controller_with_fixed_arc2(self, batch, map_fn, shuffle_batch): #, _print1, _print2):
    #     mock_images = {
    #         "train": np.zeros((45000, 3, 32, 32)),
    #         "valid": None,
    #         "test": np.zeros((45000, 1, 1, 1))
    #     }
    #     mock_labels = {
    #         "train": np.zeros((45000), dtype=np.int32),
    #         "test": np.zeros((45000), dtype=np.int32)
    #     }
    #     shuffle_batch.return_value = (mock_images['train'], mock_labels['train'])
    #     map_fn.return_value = mock_images['train']
    #     batch.return_value = (mock_images['train'], mock_labels['train'])
    #     fixed_arc = "0 3 0 0 1 0"
    #     oldconv2d = tf.nn.conv2d
    #     def newconv2d(images, w, _1, _2, data_format="NHWC"):
    #         return oldconv2d(np.zeros((45000, 32, 32, 3), dtype=np.float32), np.zeros((3, 3, 3, 24), dtype=np.float32), [1, 1, 1, 1], "SAME", data_format="NHWC")
    #     with patch.dict(tf.nn.__dict__, {'conv2d': newconv2d}):
    #         gc = GeneralChild(mock_images, mock_labels, whole_channels=True, fixed_arc=fixed_arc, keep_prob=0.9, data_format="NCHW")
    #         controller = mock.MagicMock()
    #         controller.sample_arc = np.array([int(x) for x in fixed_arc.split(" ") if x])
    #         print(f"tf_variables = {[var for var in tf.compat.v1.trainable_variables()]}")
    #         gc.connect_controller(controller)
