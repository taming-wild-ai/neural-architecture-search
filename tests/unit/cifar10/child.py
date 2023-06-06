import unittest
import unittest.mock as mock
from unittest.mock import patch

from absl import flags
flags.FLAGS(['test'])
import tensorflow as tf
import numpy as np

from src.cifar10.child import Child

class TestChild(unittest.TestCase):
    @patch('src.cifar10.child.fw.Dataset')
    @patch('src.cifar10.child.print')
    def test_constructor_nhwc(self, print, ds_ctor):
        x_train = {
            "train": np.ndarray((1, 32, 32, 3)),
            "valid": np.ndarray((1, 32, 32, 3)),
            "test": np.ndarray((1, 32, 32, 3)) }
        y_train = { "train": [1], "valid": [2], "test": [3] }
        m = Child(x_train, y_train, optim_algo="sgd")
        print.assert_any_call('-' * 80)
        print.assert_any_call("Build model generic_model")
        print.assert_any_call("Build data ops")
        ds_ctor.assert_any_call(x_train['train'], y_train['train'])
        ds_ctor().shuffle.assert_called_with(50000, m.seed)
        ds_ctor().shuffle().map.assert_called()
        ds_ctor().shuffle().map().batch.assert_called_with(m.batch_size)

    @patch('src.cifar10.child.fw.Dataset')
    @patch('src.cifar10.child.print')
    def test_constructor_nchw(self, print, ds_ctor):
        x_train = {
            "train": np.ndarray((1, 32, 32, 3)),
            "valid": np.ndarray((1, 32, 32, 3)),
            "test": np.ndarray((1, 32, 32, 3)) }
        y_train = { "train": [1], "valid": [2], "test": [3] }
        from src.cifar10.child import fw
        flags.FLAGS.data_format = "NCHW"
        m = Child(x_train, y_train, optim_algo="sgd")
        print.assert_any_call('-' * 80)
        print.assert_any_call("Build model generic_model")
        print.assert_any_call("Build data ops")
        ds_ctor.assert_any_call(x_train['train'], y_train['train'])
        ds_ctor().shuffle.assert_called_with(50000, m.seed)
        ds_ctor().shuffle().map.assert_called()
        ds_ctor().shuffle().map().batch.assert_called_with(m.batch_size)

    @patch('src.cifar10.child.fw.Dataset')
    @patch('src.cifar10.child.print')
    def test_eval_once_test(self, print, ds_ctor):
        x_train = {
            "train": np.ndarray((1, 32, 32, 3)),
            "valid": np.ndarray((1, 32, 32, 3)),
            "test": np.ndarray((1, 32, 32, 3)) }
        y_train = { "train": [1], "valid": [2], "test": [3] }
        m = Child(x_train, y_train, optim_algo="sgd")
        print.assert_any_call('-' * 80)
        print.assert_any_call("Build model generic_model")
        print.assert_any_call("Build data ops")
        ds_ctor.assert_any_call(x_train['train'], y_train['train'])
        ds_ctor().shuffle.assert_called_with(50000, None)
        ds_ctor().shuffle().map.assert_called()
        ds_ctor().shuffle().map().batch.assert_called_with(m.batch_size)
        m.global_step = "global_step"
        m.test_acc = mock.MagicMock("test_acc")
        m.eval_once("test", 'logits')

    @patch('src.cifar10.child.fw.Dataset')
    @patch('src.cifar10.child.print')
    def test_eval_once_valid(self, print, ds_ctor):
        x_train = {
            "train": np.ndarray((1, 32, 32, 3)),
            "valid": np.ndarray((1, 32, 32, 3)),
            "test": np.ndarray((1, 32, 32, 3)) }
        y_train = { "train": [1], "valid": [2], "test": [3] }
        m = Child(x_train, y_train, optim_algo="sgd")
        print.assert_any_call('-' * 80)
        print.assert_any_call("Build model generic_model")
        print.assert_any_call("Build data ops")
        ds_ctor.assert_any_call(x_train['train'], y_train['train'])
        ds_ctor().shuffle.assert_called_with(50000, None)
        ds_ctor().shuffle().map.assert_called()
        ds_ctor().shuffle().map().batch.assert_called_with(m.batch_size)
        m.global_step = "global_step"
        m.valid_acc = mock.MagicMock(return_value=4)
        m.eval_once("valid", 'logits')
