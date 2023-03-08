import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from src.cifar10.micro_child import MicroChild
from src.cifar10.macro_child import MacroChild
from src.cifar10.macro_controller import DEFINE_boolean # for controller_search_whole_channels

import src.framework as fw
import sys

class RestoreFLAGS:
    fw.FLAGS(sys.argv) # Need to parse flags before accessing them

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.old_values = {}

    def __enter__(self):
        for key, value in self.kwargs.items():
            self.old_values[key] = fw.FLAGS.__getattr__(key)
            fw.FLAGS.__setattr__(key, value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self.old_values.items():
            fw.FLAGS.__setattr__(key, value)


class TestParameterCounts(unittest.TestCase):
    @patch('src.cifar10.child.print')
    @patch('src.cifar10.macro_child.print')
    def test_macro_final_params(self, print, print2):
        IMAGES = {
            'train': np.ndarray((45000, 32, 32, 3), dtype=np.float32),
            'valid': np.ndarray((5000, 32, 32, 3), dtype=np.float32),
            'test':  np.ndarray((5000, 32, 32, 3), dtype=np.float32)}
        LABELS = {
            'train': np.ndarray((45000), dtype=np.int32),
            'valid': np.ndarray((5000), dtype=np.int32),
            'test':  np.ndarray((5000), dtype=np.int32)}
        with RestoreFLAGS(batch_size=100, child_out_filters=96, data_format="NCHW", num_epochs=310, child_num_layers=24, child_keep_prob=0.5, child_use_aux_heads=True):
            with tf.Graph().as_default():
                mc = MacroChild(
                    IMAGES,
                    LABELS,
                    optim_algo="momentum")
                mc.whole_channels = True
                fixed_arc = "0"
                fixed_arc +=" 3 0"
                fixed_arc +=" 0 1 0"
                fixed_arc +=" 2 0 0 1"
                fixed_arc +=" 2 0 0 0 0"
                fixed_arc +=" 3 1 1 0 1 0"
                fixed_arc +=" 2 0 0 0 0 0 1"
                fixed_arc +=" 2 0 1 1 0 1 1 1"
                fixed_arc +=" 1 0 1 1 1 0 1 0 1"
                fixed_arc +=" 0 0 0 0 0 0 0 0 0 0"
                fixed_arc +=" 2 0 0 0 0 0 1 0 0 0 0"
                fixed_arc +=" 0 1 0 0 1 1 0 0 0 0 1 1"
                fixed_arc +=" 2 0 1 0 0 0 0 0 1 0 1 1 0"
                fixed_arc +=" 1 0 0 1 0 0 0 1 1 1 0 1 0 1"
                fixed_arc +=" 0 1 1 0 1 0 1 0 0 0 0 0 1 0 0"
                fixed_arc +=" 2 0 0 1 0 0 0 0 0 0 0 1 0 1 0 1"
                fixed_arc +=" 2 0 1 0 0 0 1 0 0 1 1 1 1 0 0 1 0"
                fixed_arc +=" 2 0 0 0 0 1 0 1 0 1 0 0 1 0 1 0 0 1"
                fixed_arc +=" 3 0 1 1 0 1 0 0 0 0 0 1 0 1 0 1 0 0 0"
                fixed_arc +=" 3 0 1 1 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1"
                fixed_arc +=" 0 1 0 0 1 0 1 1 0 0 0 1 0 0 0 0 0 1 1 0 0"
                fixed_arc +=" 3 0 1 0 1 1 0 0 1 0 1 1 0 1 1 0 1 0 0 1 0 0"
                fixed_arc +=" 0 1 0 1 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0"
                fixed_arc +=" 0 1 1 0 0 0 1 1 1 0 1 0 0 0 1 0 1 0 0 1 1 0 0 0"
                mc.fixed_arc = fixed_arc
                mc.sample_arc = np.array([int(x) for x in fixed_arc.split(" ") if x])
                mc._build_train()
                print2.assert_any_call("-" * 80)
                print2.assert_any_call("Build model child")
                print2.assert_any_call("Build data ops")
                print.assert_called_with("Model has 42559392 params")

    @patch('src.cifar10.child.print')
    @patch('src.cifar10.macro_child.print')
    def test_macro_search_params(self, print, print2):
        IMAGES = {
            'train': np.ndarray((45000, 32, 32, 3), dtype=np.float32),
            'valid': np.ndarray((5000, 32, 32, 3), dtype=np.float32),
            'test':  np.ndarray((5000, 32, 32, 3), dtype=np.float32)}
        LABELS = {
            'train': np.ndarray((45000), dtype=np.int32),
            'valid': np.ndarray((5000), dtype=np.int32),
            'test':  np.ndarray((5000), dtype=np.int32)}
        with RestoreFLAGS(batch_size=128, child_out_filters=36, data_format="NCHW", num_epochs=310, child_num_layers=14, child_keep_prob=0.9, child_use_aux_heads=True):
            with tf.Graph().as_default():
                mc = MacroChild(
                    IMAGES,
                    LABELS,
                    optim_algo="momentum")
                mc.whole_channels = True
                mc.fixed_arc = None
                fixed_arc = "0"
                fixed_arc +=" 3 0"
                fixed_arc +=" 0 1 0"
                fixed_arc +=" 2 0 0 1"
                fixed_arc +=" 2 0 0 0 0"
                fixed_arc +=" 3 1 1 0 1 0"
                fixed_arc +=" 2 0 0 0 0 0 1"
                fixed_arc +=" 2 0 1 1 0 1 1 1"
                fixed_arc +=" 1 0 1 1 1 0 1 0 1"
                fixed_arc +=" 0 0 0 0 0 0 0 0 0 0"
                fixed_arc +=" 2 0 0 0 0 0 1 0 0 0 0"
                fixed_arc +=" 0 1 0 0 1 1 0 0 0 0 1 1"
                fixed_arc +=" 2 0 1 0 0 0 0 0 1 0 1 1 0"
                fixed_arc +=" 1 0 0 1 0 0 0 1 1 1 0 1 0 1"
                fixed_arc +=" 0 1 1 0 1 0 1 0 0 0 0 0 1 0 0"
                fixed_arc +=" 2 0 0 1 0 0 0 0 0 0 0 1 0 1 0 1"
                fixed_arc +=" 2 0 1 0 0 0 1 0 0 1 1 1 1 0 0 1 0"
                fixed_arc +=" 2 0 0 0 0 1 0 1 0 1 0 0 1 0 1 0 0 1"
                fixed_arc +=" 3 0 1 1 0 1 0 0 0 0 0 1 0 1 0 1 0 0 0"
                fixed_arc +=" 3 0 1 1 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1"
                fixed_arc +=" 0 1 0 0 1 0 1 1 0 0 0 1 0 0 0 0 0 1 1 0 0"
                fixed_arc +=" 3 0 1 0 1 1 0 0 1 0 1 1 0 1 1 0 1 0 0 1 0 0"
                fixed_arc +=" 0 1 0 1 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0"
                fixed_arc +=" 0 1 1 0 0 0 1 1 1 0 1 0 0 0 1 0 1 0 0 1 1 0 0 0"
                mc.sample_arc = np.array([int(x) for x in fixed_arc.split(" ") if x])
                mc._build_train()
                print2.assert_any_call("-" * 80)
                print2.assert_any_call("Build model child")
                print2.assert_any_call("Build data ops")
                print.assert_called_with("Model has 810756 params")

    @patch('src.cifar10.child.print')
    @patch('src.cifar10.micro_child.print')
    def test_micro_final_params(self, print, print2):
        IMAGES = {
            'train': np.ndarray((45000, 32, 32, 3), dtype=np.float32),
            'valid': np.ndarray((5000, 32, 32, 3), dtype=np.float32),
            'test':  np.ndarray((5000, 32, 32, 3), dtype=np.float32)}
        LABELS = {
            'train': np.ndarray((45000), dtype=np.int32),
            'valid': np.ndarray((5000), dtype=np.int32),
            'test':  np.ndarray((5000), dtype=np.int32)}
        with RestoreFLAGS(batch_size=144, child_out_filters=36, data_format="NCHW", child_num_aggregate=None, child_num_branches=5, child_num_cells=5, num_epochs=630, child_num_layers=15, child_use_aux_heads=True):
            with tf.Graph().as_default():
                mc = MicroChild(
                    IMAGES,
                    LABELS,
                    optim_algo="adam")
                mc.fixed_arc = "0 2 0 0 0 4 0 1 0 4 1 1 1 0 0 1 0 2 1 1 1 0 1 0 0 3 0 2 1 1 3 1 1 0 0 4 0 3 1 1"
                fixed_arc = np.array([int(x) for x in mc.fixed_arc.split(" ") if x])
                mc.normal_arc = fixed_arc[:4 * mc.num_cells]
                mc.reduce_arc = fixed_arc[4 * mc.num_cells:]
                mc._build_train()
                print2.assert_any_call("-" * 80)
                print2.assert_any_call("Build model child")
                print2.assert_any_call("Build data ops")
                for layer_id in range(5):
                    print.assert_any_call(f'Layer  {layer_id}: Tensor("child/layer_{layer_id}/final_combine/concat:0", shape=(None, 180, 32, 32), dtype=float32)')
                print.assert_any_call('Layer  5: Tensor("child/layer_5/final_combine/concat:0", shape=(None, 288, 16, 16), dtype=float32)')
                for layer_id in range(6, 10):
                    print.assert_any_call(f'Layer  {layer_id}: Tensor("child/layer_{layer_id}/final_combine/concat:0", shape=(None, 360, 16, 16), dtype=float32)')
                print.assert_any_call('Layer 10: Tensor("child/layer_10/final_combine/concat:0", shape=(None, 360, 16, 16), dtype=float32)')
                print.assert_any_call('Layer 11: Tensor("child/layer_11/final_combine/concat:0", shape=(None, 576, 8, 8), dtype=float32)')
                for layer_id in range(12, 17):
                    print.assert_any_call(f'Layer {layer_id}: Tensor("child/layer_{layer_id}/final_combine/concat:0", shape=(None, 720, 8, 8), dtype=float32)')
                print.assert_any_call("Aux head uses 494848 params")
                print.assert_any_call("Model has 3894372 params")

    @patch('src.cifar10.child.print')
    @patch('src.cifar10.micro_child.print')
    def test_micro_search_params(self, print, print2):
        IMAGES = {
            'train': np.ndarray((45000, 32, 32, 3), dtype=np.float32),
            'valid': np.ndarray((5000, 32, 32, 3), dtype=np.float32),
            'test':  np.ndarray((5000, 32, 32, 3), dtype=np.float32)}
        LABELS = {
            'train': np.ndarray((45000), dtype=np.int32),
            'valid': np.ndarray((5000), dtype=np.int32),
            'test':  np.ndarray((5000), dtype=np.int32)}
        with RestoreFLAGS(batch_size=160, child_out_filters=20, data_format="NCHW", child_num_aggregate=None, child_num_branches=5, child_num_cells=5, num_epochs=150, child_num_layers=6, child_use_aux_heads=True):
            with tf.Graph().as_default():
                mc = MicroChild(
                    IMAGES,
                    LABELS,
                    optim_algo="adam")
                fixed_arc="0 2 0 0 0 4 0 1 0 4 1 1 1 0 0 1 0 2 1 1 1 0 1 0 0 3 0 2 1 1 3 1 1 0 0 4 0 3 1 1"
                fixed_arc = np.array([int(x) for x in fixed_arc.split(" ") if x])
                mc.normal_arc = fixed_arc[:4 * mc.num_cells]
                mc.reduce_arc = fixed_arc[4 * mc.num_cells:]
                mc._build_train()
                for layer_num in range(8):
                    if 2 > layer_num:
                        expected_shape = (None, 20, 32, 32)
                    elif 5 > layer_num:
                        expected_shape = (None, 40, 16, 16)
                    else:
                        expected_shape = (None, 80, 8, 8)
                    print.assert_any_call(f'Layer  {layer_num}: Tensor("child/layer_{layer_num}/Reshape_2:0", shape={expected_shape}, dtype=float32)')
                print2.assert_any_call("-" * 80)
                print2.assert_any_call("Build model child")
                print2.assert_any_call("Build data ops")
                print.assert_any_call("Aux head uses 412928 params")
                print.assert_called_with("Model has 5373140 params")


if "__main__" == __name__:
    unittest.main()