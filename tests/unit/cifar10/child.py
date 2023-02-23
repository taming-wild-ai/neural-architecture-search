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
                    m = Child(x_train, y_train)
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
                    m = Child(x_train, y_train)
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
                    m = Child(x_train, y_train)
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
                    m = Child(x_train, y_train)
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

    @patch('src.cifar10.child.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.child.get_train_ops', return_value=(1, 2, 3, 4))
    @patch('src.cifar10.child.fw.equal', return_value="equal")
    @patch('src.cifar10.child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.child.fw.reduce_mean', return_value="reduce_mean")
    @patch('src.cifar10.child.fw.sparse_softmax_cross_entropy_with_logits', return_value="sscewl")
    @patch('src.cifar10.child.print')
    def test_build_train(self, print, sscewl, reduce_mean, argmax, equal, get_train_ops, to_int32, reduce_sum):
        x_train = {
            "train": np.ndarray((1, 32, 32, 3)),
            "valid": np.ndarray((1, 32, 32, 3)),
            "test": np.ndarray((1, 32, 32, 3)) }
        y_train = { "train": [1], "valid": [2], "test": [3] }
        with patch('src.cifar10.child.fw.shuffle_batch', return_value=(x_train, y_train)) as sb:
            with patch('src.cifar10.child.fw.batch', return_value=(x_train, y_train)) as batch:
                with patch('src.cifar10.child.fw.map_fn', return_value=x_train) as map_fn:
                    m = Child(x_train, y_train)
                    with patch.object(m, '_model', return_value="model") as model:
                        print.assert_any_call('-' * 80)
                        print.assert_any_call("Build model generic_model")
                        print.assert_any_call("Build data ops")
                        map_fn.assert_called()
                        m._build_train()
                        print.assert_any_call('Build train graph')
                        sscewl.assert_called_with(logits='model', labels={'train': [1], 'valid': [2], 'test': [3], 'valid_original': [2]})
                        reduce_mean.assert_called_with('sscewl')
                        argmax.assert_called_with('model', axis=1)
                        equal.assert_called()
                        get_train_ops.assert_called()
                        to_int32.assert_called_with('equal')
                        reduce_sum.assert_called_with('to_int32')

    @patch('src.cifar10.child.fw.equal', return_value="equal")
    @patch('src.cifar10.child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.child.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.child.print')
    def test_build_valid(self, print, to_int32, reduce_sum, argmax, equal):
        x_train = {
            "train": np.ndarray((1, 32, 32, 3)),
            "valid": np.ndarray((1, 32, 32, 3)),
            "test": np.ndarray((1, 32, 32, 3)) }
        y_train = { "train": [1], "valid": [2], "test": [3] }
        with patch('src.cifar10.child.fw.shuffle_batch', return_value=(x_train, y_train)) as sb:
            with patch('src.cifar10.child.fw.batch', return_value=(x_train, y_train)) as batch:
                with patch('src.cifar10.child.fw.map_fn', return_value=x_train) as map_fn:
                    m = Child(x_train, y_train)
                    with patch.object(m, '_model', return_value="model") as model:
                        print.assert_any_call('-' * 80)
                        print.assert_any_call("Build model generic_model")
                        print.assert_any_call("Build data ops")
                        map_fn.assert_called()
                        m._build_valid()
                        print.assert_any_call('Build valid graph')
                        to_int32.assert_called_with("equal")
                        reduce_sum.assert_called_with('to_int32')
                        argmax.assert_called_with('model', axis=1)
                        equal.assert_called()

    @patch('src.cifar10.child.fw.equal', return_value="equal")
    @patch('src.cifar10.child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.child.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.child.print')
    def test_build_valid_rl(self, print, to_int32, reduce_sum, argmax, equal):
        x_train = {
            "train": np.ndarray((1, 32, 32, 3)),
            "valid": np.ndarray((1, 32, 32, 3)),
            "test": np.ndarray((1, 32, 32, 3)) }
        y_train = { "train": [1], "valid": [2], "test": [3] }
        with patch('src.cifar10.child.fw.shuffle_batch', return_value=(x_train, y_train)) as sb:
            with patch('src.cifar10.child.fw.batch', return_value=(x_train, y_train)) as batch:
                with patch('src.cifar10.child.fw.map_fn', return_value=x_train) as map_fn:
                    m = Child(x_train, y_train)
                    with patch.object(m, '_model', return_value="model") as model:
                        print.assert_any_call('-' * 80)
                        print.assert_any_call("Build model generic_model")
                        print.assert_any_call("Build data ops")
                        map_fn.assert_called()
                        m.build_valid_rl()
                        print.assert_any_call('Build valid graph on shuffled data')
                        to_int32.assert_called_with("equal")
                        reduce_sum.assert_called_with('to_int32')
                        argmax.assert_called_with('model', axis=1)
                        equal.assert_called()

    @patch('src.cifar10.child.fw.equal', return_value="equal")
    @patch('src.cifar10.child.fw.argmax', return_value="argmax")
    @patch('src.cifar10.child.fw.reduce_sum', return_value="reduce_sum")
    @patch('src.cifar10.child.fw.to_int32', return_value="to_int32")
    @patch('src.cifar10.child.print')
    def test_build_test(self, print, to_int32, reduce_sum, argmax, equal):
        x_train = {
            "train": np.ndarray((1, 32, 32, 3)),
            "valid": np.ndarray((1, 32, 32, 3)),
            "test": np.ndarray((1, 32, 32, 3)) }
        y_train = { "train": [1], "valid": [2], "test": [3] }
        with patch('src.cifar10.child.fw.shuffle_batch', return_value=(x_train, y_train)) as sb:
            with patch('src.cifar10.child.fw.batch', return_value=(x_train, y_train)) as batch:
                with patch('src.cifar10.child.fw.map_fn', return_value=x_train) as map_fn:
                    m = Child(x_train, y_train)
                    with patch.object(m, '_model', return_value="model") as model:
                        print.assert_any_call('-' * 80)
                        print.assert_any_call("Build model generic_model")
                        print.assert_any_call("Build data ops")
                        map_fn.assert_called()
                        m._build_test()
                        print.assert_any_call('Build test graph')
                        to_int32.assert_called_with("equal")
                        reduce_sum.assert_called_with('to_int32')
                        argmax.assert_called_with('model', axis=1)
                        equal.assert_called()
