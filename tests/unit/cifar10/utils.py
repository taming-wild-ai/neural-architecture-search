import sys
import unittest
import unittest.mock as mock
from unittest.mock import patch

from src.utils import print_user_flags, Logger, get_train_ops
from src.cifar10.child import ClipMode, Optimizer, LearningRate

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class TestUtils(unittest.TestCase):
    @patch('src.utils.print')
    def test_print_user_flags(self, print):
        global user_flags
        user_flags = ['flag1', 'flag2']
        print_user_flags()
        print.assert_any_call('-' * 80)

    @patch('src.utils.zip', return_value=[[mock.MagicMock(), 2]])
    @patch('src.utils.fw.exp_decay', return_value="exp_decay")
    @patch('src.utils.fw.maximum', return_value="maximum")
    @patch('src.utils.fw.sqrt', return_value=0.0)
    @patch('src.utils.fw.global_norm', return_value="global_norm")
    @patch('src.utils.fw.gradients', return_value=[0.0])
    @patch('src.utils.fw.add_n', return_value=1.0)
    @patch('src.utils.fw.reduce_sum')
    def test_get_train_ops_grad_norms_momentum_global(self, reduce_sum, add_n, gradients, global_norm, sqrt, max, exp_decay, zip):
        var = tf.ones((1))
        mock_momentum = mock.MagicMock()
        with patch('src.cifar10.child.fw.Optimizer.Momentum', return_value=mock_momentum) as mom:
            get_train_ops(0.0, [var], 0, LearningRate.new(False, 0.1, 2, 10000, 0.1, 5, 6, 7, 8, 9), get_grad_norms=True, optim_algo=Optimizer.new("momentum", False, 1, 1), clip_mode=ClipMode.new("global", 0.0), num_train_batches=1)
            add_n.assert_called_with([reduce_sum()])
            gradients.assert_called_with(1e-4, [var])
            global_norm.assert_called_with([0.0])
            sqrt.assert_called_with(reduce_sum())
            max.assert_called_with('exp_decay', 5)
            exp_decay.assert_called_with(0.1, 'maximum', 10000, 0.1, staircase=True)
            mom.assert_called_with('maximum')
            mock_momentum.apply_gradients.assert_called_with(zip.return_value, global_step=0)

    @patch('src.utils.zip', return_value=[[mock.MagicMock(), 2]])
    @patch('src.utils.fw.exp_decay', return_value="exp_decay")
    @patch('src.utils.fw.maximum', return_value="maximum")
    @patch('src.utils.fw.sqrt', return_value=0.0)
    @patch('src.utils.fw.global_norm', return_value="global_norm")
    @patch('src.utils.fw.gradients', return_value=[0.0])
    @patch('src.utils.fw.add_n', return_value=1.0)
    @patch('src.utils.fw.reduce_sum')
    def test_get_train_ops_grad_norms_adam_global(self, reduce_sum, add_n, gradients, global_norm, sqrt, max, exp_decay, zip):
        mock_adam = mock.MagicMock()
        with patch('src.utils.fw.Optimizer.Adam', return_value=mock_adam) as mom:
            var = tf.ones((1))
            get_train_ops(0.0, [var], 0, LearningRate.new(False, 0.1, 2, 10000, 0.1, 5, 6, 7, 8, 9), get_grad_norms=True, optim_algo=Optimizer.new("adam", False, 1, 1), clip_mode=ClipMode.new("global", 0.0), num_train_batches=1)
            add_n.assert_called_with([reduce_sum()])
            gradients.assert_called_with(1e-4, [var])
            global_norm.assert_called_with([0.0])
            sqrt.assert_called_with(reduce_sum())
            max.assert_called_with('exp_decay', 5)
            exp_decay.assert_called_with(0.1, 'maximum', 10000, 0.1, staircase=True)
            mom.assert_called_with('maximum')
            mock_adam.apply_gradients.assert_called_with(zip.return_value, global_step=0)

    @patch('src.utils.fw.cond')
    @patch('src.utils.fw.greater_equal', return_value="ge")
    @patch('src.utils.fw.less')
    @patch('src.utils.zip', return_value=[[mock.MagicMock(), 2]])
    @patch('src.utils.fw.exp_decay', return_value="exp_decay")
    @patch('src.utils.fw.maximum', return_value="maximum")
    @patch('src.utils.fw.sqrt', return_value=0.0)
    @patch('src.utils.fw.global_norm', return_value="global_norm")
    @patch('src.utils.fw.gradients', return_value=[0.0])
    @patch('src.utils.fw.add_n', return_value=1.0)
    @patch('src.utils.fw.reduce_sum')
    def test_get_train_ops_no_grad_norms_sgd_norm(self, reduce_sum, add_n, gradients, global_norm, sqrt, max, exp_decay, zip, less, ge, cond):
        mock_sro = mock.MagicMock(name='sro')
        mock_sgd = mock.MagicMock(name='sgd')
        with patch('src.utils.fw.Optimizer.SGD', return_value=mock_sgd) as sgd:
            with patch(
                'src.utils.fw.Optimizer.SyncReplicas',
                return_value=mock_sro) as sro:
                var = tf.ones((1))
                get_train_ops(
                    0.0,
                    [var],
                    0,
                    LearningRate.new(True, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                    optim_algo=Optimizer.new("sgd", True, 1, 1),
                    clip_mode=ClipMode.new("norm", 0.0),
                    num_train_batches=1,
                    get_grad_norms=True)
                add_n.assert_called_with([reduce_sum()])
                gradients.assert_called_with(1e-4, [var])
                global_norm.assert_called_with([0.0])
                sqrt.assert_called_with(reduce_sum())
                max.assert_not_called()
                exp_decay.assert_not_called()
                zip.assert_called_with([var], [0.0])
                less.assert_not_called()
                ge.assert_called()
                cond.assert_called()
                sgd.assert_called_with(cond())
                sro.assert_called_with(mock_sgd, 1, 1)
                mock_sgd.apply_gradients.assert_not_called()
                mock_sro.apply_gradients.assert_called_with(
                    zip.return_value,
                    global_step=0)


class TestLogger(unittest.TestCase):
    @patch('src.utils.open', return_value="open")
    def test_constructor(self, open_mock):
        logger = Logger('output.txt')
        self.assertEqual(sys.stdout, logger.terminal)
        self.assertEqual(logger.log, "open")
        open_mock.assert_called_with("output.txt", "a")

    @patch('src.utils.open', return_value="open")
    def test_write(self, open_mock):
        logger = Logger('output.txt')
        logger.terminal = mock.MagicMock()
        logger.log = mock.MagicMock()
        logger.write("message")
        logger.terminal.write.assert_called_with("message")
        logger.terminal.flush.assert_called_with()
        logger.log.write.assert_called_with("message")
        logger.log.flush.assert_called_with()

if "__main__" == __name__:
    unittest.main()