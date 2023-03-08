import sys
import unittest
import unittest.mock as mock
from unittest.mock import patch

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from src.utils import print_user_flags, Logger, user_flags, get_train_ops
from src.cifar10.child import ClipMode, Optimizer, LearningRate

class TestUtils(unittest.TestCase):
    @patch('src.utils.print')
    def test_print_user_flags(self, print):
        global user_flags
        user_flags = ['flag1', 'flag2']
        print_user_flags()
        print.assert_any_call('-' * 80)

    @patch('src.utils.fw.exp_decay', return_value="exp_decay")
    @patch('src.utils.fw.maximum', return_value="maximum")
    @patch('src.utils.fw.sqrt', return_value=0.0)
    @patch('src.utils.fw.global_norm', return_value="global_norm")
    @patch('src.utils.fw.gradients', return_value=[0.0])
    @patch('src.utils.fw.add_n', return_value=1.0)
    def test_get_train_ops_grad_norms_momentum_global(self, add_n, gradients, global_norm, sqrt, max, exp_decay):
        mock_momentum = mock.MagicMock()
        with patch('src.utils.fw.Optimizer.Momentum', return_value=mock_momentum) as mom:
            var = tf.ones((1))
            get_train_ops(0.0, [var], 0, LearningRate.new(False, 0.1, 2, 10000, 0.1, 5, 6, 7, 8, 9), get_grad_norms=True, optim_algo=Optimizer.new("momentum", False, 1, 1), clip_mode=ClipMode.new("global", 0.0), num_train_batches=1)
            add_n.assert_called()
            gradients.assert_called_with(1e-4, [var])
            global_norm.assert_called_with([0.0])
            sqrt.assert_called()
            max.assert_called_with('exp_decay', 5)
            exp_decay.assert_called_with(0.1, 'maximum', 10000, 0.1, staircase=True)
            mom.assert_called_with('maximum')
            mock_momentum.apply_gradients.assert_called()

    @patch('src.utils.fw.exp_decay', return_value="exp_decay")
    @patch('src.utils.fw.maximum', return_value="maximum")
    @patch('src.utils.fw.sqrt', return_value=0.0)
    @patch('src.utils.fw.global_norm', return_value="global_norm")
    @patch('src.utils.fw.gradients', return_value=[0.0])
    @patch('src.utils.fw.add_n', return_value=1.0)
    def test_get_train_ops_grad_norms_adam_global(self, add_n, gradients, global_norm, sqrt, max, exp_decay):
        mock_adam = mock.MagicMock()
        with patch('src.utils.fw.Optimizer.Adam', return_value=mock_adam) as mom:
            var = tf.ones((1))
            get_train_ops(0.0, [var], 0, LearningRate.new(False, 0.1, 2, 10000, 0.1, 5, 6, 7, 8, 9), get_grad_norms=True, optim_algo=Optimizer.new("adam", False, 1, 1), clip_mode=ClipMode.new("global", 0.0), num_train_batches=1)
            add_n.assert_called()
            gradients.assert_called_with(1e-4, [var])
            global_norm.assert_called_with([0.0])
            sqrt.assert_called()
            max.assert_called_with('exp_decay', 5)
            exp_decay.assert_called_with(0.1, 'maximum', 10000, 0.1, staircase=True)
            mom.assert_called_with('maximum')
            mock_adam.apply_gradients.assert_called()

    @patch('src.utils.fw.to_float', return_value=1.0)
    @patch('src.utils.fw.exp_decay', return_value="exp_decay")
    @patch('src.utils.fw.maximum', return_value="maximum")
    @patch('src.utils.fw.sqrt', return_value=0.0)
    @patch('src.utils.fw.global_norm', return_value="global_norm")
    @patch('src.utils.fw.gradients', return_value=[0.0])
    @patch('src.utils.fw.add_n', return_value=1.0)
    def test_get_train_ops_no_grad_norms_sgd_norm(self, add_n, gradients, global_norm, sqrt, max, exp_decay, to_float):
        mock_sro = mock.MagicMock()
        mock_sgd = mock.MagicMock()
        with patch('src.utils.fw.Optimizer.SGD', return_value=mock_sgd) as sgd:
            with patch('src.utils.fw.Optimizer.SyncReplicas', return_value=mock_sro) as sro:
                var = tf.ones((1))
                get_train_ops(0.0, [var], 0, LearningRate.new(True, 1, 2, 3, 4, 5, 6, 7, 8, 9), optim_algo=Optimizer.new("sgd", True, 1, 1), clip_mode=ClipMode.new("norm", 0.0), num_train_batches=1)
                add_n.assert_called()
                gradients.assert_called_with(1e-4, [var])
                global_norm.assert_called_with([0.0])
                sqrt.assert_called()
                max.assert_not_called()
                exp_decay.assert_not_called()
                sgd.assert_called()
                sro.assert_called()
                mock_sgd.apply_gradients.assert_not_called()
                mock_sro.apply_gradients.assert_called()
                to_float.assert_called()


class TestLogger(unittest.TestCase):
    @patch('src.utils.open', return_value="open")
    def test_constructor(self, open_mock):
        l = Logger('output.txt')
        self.assertEqual(sys.stdout, l.terminal)
        self.assertEqual(l.log, "open")
        open_mock.assert_called_with("output.txt", "a")

    @patch('src.utils.open', return_value="open")
    def test_write(self, open_mock):
        l = Logger('output.txt')
        l.terminal = mock.MagicMock()
        l.log = mock.MagicMock()
        l.write("message")
        l.terminal.write.assert_called_with("message")
        l.terminal.flush.assert_called_with()
        l.log.write.assert_called_with("message")
        l.log.flush.assert_called_with()

if "__main__" == __name__:
    unittest.main()