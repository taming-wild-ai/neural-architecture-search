import unittest
import unittest.mock as mock
from unittest.mock import patch

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

from src.cifar10.child import Child

class TestChild(unittest.TestCase):
    @patch('src.cifar10.child.print')
    def test_constructor_nhwc(self, print):
        x_train = {
            "train": np.ndarray((1, 32, 32, 3)),
            "valid": np.ndarray((1, 32, 32, 3)),
            "test": np.ndarray((1, 32, 32, 3)) }
        y_train = { "train": [1], "valid": [2], "test": [3] }
        with patch('src.cifar10.child.fw.shuffle_batch', return_value=(x_train, y_train)) as sb:
            with patch('src.cifar10.child.fw.batch', return_value=(x_train, y_train)) as batch:
                with patch('src.cifar10.child.fw.map_fn', return_value=x_train) as map_fn:
                    m = Child(x_train, y_train, optim_algo="sgd")
                    print.assert_any_call('-' * 80)
                    print.assert_any_call("Build model generic_model")
                    print.assert_any_call("Build data ops")
                    map_fn.assert_called()

    @patch('src.cifar10.child.print')
    def test_constructor_nchw(self, print):
        x_train = {
            "train": np.ndarray((1, 32, 32, 3)),
            "valid": np.ndarray((1, 32, 32, 3)),
            "test": np.ndarray((1, 32, 32, 3)) }
        y_train = { "train": [1], "valid": [2], "test": [3] }
        with patch('src.cifar10.child.fw.shuffle_batch', return_value=(x_train, y_train)) as sb:
            with patch('src.cifar10.child.fw.batch', return_value=(x_train, y_train)) as batch:
                with patch('src.cifar10.child.fw.map_fn', return_value=x_train) as map_fn:
                    from src.cifar10.child import fw
                    fw.FLAGS.data_format = "NCHW"
                    m = Child(x_train, y_train, optim_algo="sgd")
                    print.assert_any_call('-' * 80)
                    print.assert_any_call("Build model generic_model")
                    print.assert_any_call("Build data ops")
                    map_fn.assert_called()

    @patch('src.cifar10.child.print')
    def test_eval_once_test(self, print):
        x_train = {
            "train": np.ndarray((1, 32, 32, 3)),
            "valid": np.ndarray((1, 32, 32, 3)),
            "test": np.ndarray((1, 32, 32, 3)) }
        y_train = { "train": [1], "valid": [2], "test": [3] }
        with patch('src.cifar10.child.fw.shuffle_batch', return_value=(x_train, y_train)) as sb:
            with patch('src.cifar10.child.fw.batch', return_value=(x_train, y_train)) as batch:
                with patch('src.cifar10.child.fw.map_fn', return_value=x_train) as map_fn:
                    m = Child(x_train, y_train, optim_algo="sgd")
                    print.assert_any_call('-' * 80)
                    print.assert_any_call("Build model generic_model")
                    print.assert_any_call("Build data ops")
                    map_fn.assert_called()
                    mock_sess = mock.MagicMock()
                    m.global_step = "global_step"
                    m.test_acc = "test_acc"
                    m.eval_once(mock_sess, "test")
                    mock_sess.run.assert_any_call("global_step")
                    mock_sess.run.assert_called_with('test_acc', feed_dict=None)

    @patch('src.cifar10.child.print')
    def test_eval_once_valid(self, print):
        x_train = {
            "train": np.ndarray((1, 32, 32, 3)),
            "valid": np.ndarray((1, 32, 32, 3)),
            "test": np.ndarray((1, 32, 32, 3)) }
        y_train = { "train": [1], "valid": [2], "test": [3] }
        with patch('src.cifar10.child.fw.shuffle_batch', return_value=(x_train, y_train)) as sb:
            with patch('src.cifar10.child.fw.batch', return_value=(x_train, y_train)) as batch:
                with patch('src.cifar10.child.fw.map_fn', return_value=x_train) as map_fn:
                    m = Child(x_train, y_train, optim_algo="sgd")
                    print.assert_any_call('-' * 80)
                    print.assert_any_call("Build model generic_model")
                    print.assert_any_call("Build data ops")
                    map_fn.assert_called()
                    mock_sess = mock.MagicMock()
                    m.global_step = "global_step"
                    m.valid_acc = "valid_acc"
                    m.eval_once(mock_sess, "valid")
                    mock_sess.run.assert_any_call("global_step")
                    mock_sess.run.assert_called_with('valid_acc', feed_dict=None)
