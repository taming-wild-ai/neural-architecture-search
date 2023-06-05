import sys
import unittest
import unittest.mock as mock
from unittest.mock import patch

from absl import flags
flags.FLAGS(['test'])

from src.utils import print_user_flags, Logger, get_train_ops
from src.cifar10.child import ClipMode, Optimizer, LearningRate
import src.framework as fw

import tensorflow as tf

class TestUtils(unittest.TestCase):
    @patch('src.utils.print')
    def test_print_user_flags(self, print):
        global user_flags
        user_flags = ['flag1', 'flag2']
        print_user_flags()
        print.assert_any_call('-' * 80)

    @patch('src.utils.zip', return_value=[[mock.MagicMock(), 2]])
    @patch('src.utils.fw.exp_decay', return_value=mock.MagicMock(return_value="exp_decay"))
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
            global_step = fw.Variable(0, dtype=fw.int64)
            train_op, lr, grad_norm, _opt, grad_norms = get_train_ops(
                global_step,
                LearningRate.new(False, 0.1, 2, 10000, 0.1, 5, 6, 7, 8, 9),
                get_grad_norms=True,
                optim_algo=Optimizer.new("momentum", False, 1, 1),
                clip_mode=ClipMode.new("global", 0.0),
                num_train_batches=1)
            lr()
            train_op(0.0, [var])
            grad_norm(0.0, [var])
            grad_norms(0.0, [var])
            add_n.assert_called_with([reduce_sum()])
            # gradients.assert_called_with(1e-4, [var])
            self.assertEqual(len(gradients.call_args_list), 1)
            self.assertEqual(1e-4, gradients.call_args_list[0][0][0])
            self.assertEqual(list, type(gradients.call_args_list[0][0][1]))
            global_norm.assert_called_with([0.0])
            sqrt.assert_called_with(reduce_sum())
            max.assert_called_with('exp_decay', 5)
            exp_decay.assert_called_with(0.1, 10000, 0.1, staircase=True)
            exp_decay().assert_called_with('maximum')
            mom.assert_called_with(lr)
            mock_momentum.apply_gradients.assert_called_with(
                zip.return_value,
                global_step=global_step)

    @patch('src.utils.zip', return_value=[[mock.MagicMock(), 2]])
    @patch('src.utils.fw.exp_decay', return_value=mock.MagicMock(return_value="exp_decay"))
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
            global_step = fw.Variable(0, dtype=fw.int32)
            train_op, lr, grad_norm, _opt, grad_norms = get_train_ops(
                global_step,
                LearningRate.new(False, 0.1, 2, 10000, 0.1, 5, 6, 7, 8, 9),
                get_grad_norms=True,
                optim_algo=Optimizer.new("adam", False, 1, 1),
                clip_mode=ClipMode.new("global", 0.0),
                num_train_batches=1)
            train_op(0.0, [var])
            lr()
            grad_norm(0.0, [var])
            grad_norms(0.0, [var])
            add_n.assert_called_with([reduce_sum()])
            # gradients.assert_called_with(1e-4, [var])
            self.assertEqual(len(gradients.call_args_list), 1)
            self.assertEqual(1e-4, gradients.call_args_list[0][0][0])
            self.assertEqual(list, type(gradients.call_args_list[0][0][1]))
            global_norm.assert_called_with([0.0])
            sqrt.assert_called_with(reduce_sum())
            max.assert_called_with('exp_decay', 5)
            exp_decay.assert_called_with(0.1, 10000, 0.1, staircase=True)
            exp_decay().assert_called_with('maximum')
            mom.assert_called_with(lr)
            mock_adam.apply_gradients.assert_called_with(
                zip.return_value,
                global_step=global_step)

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
                global_step = fw.Variable(0, dtype=fw.int64)
                train_op, lr, grad_norm, _opt, grad_norms = get_train_ops(
                    global_step,
                    LearningRate.new(True, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                    optim_algo=Optimizer.new("sgd", True, 1, 1),
                    clip_mode=ClipMode.new("norm", 0.0),
                    num_train_batches=1,
                    get_grad_norms=True)
                train_op(0.0, [var])
                lr()
                grad_norm(0.0, [var])
                grad_norms(0.0, [var])
                add_n.assert_called_with([reduce_sum()])
                # gradients.assert_called_with(1e-4, [var])
                self.assertEqual(len(gradients.call_args_list), 1)
                self.assertEqual(1e-4, gradients.call_args_list[0][0][0])
                self.assertEqual(list, type(gradients.call_args_list[0][0][1]))
                global_norm.assert_called_with([0.0])
                sqrt.assert_called_with(reduce_sum())
                max.assert_not_called()
                exp_decay.assert_not_called()
                zip.assert_any_call([var], [0.0])
                less.assert_not_called()
                ge.assert_called()
                cond.assert_called()
                sgd.assert_called_with(lr)
                sro.assert_called_with(mock_sgd, 1, 1)
                mock_sgd.apply_gradients.assert_not_called()
                mock_sro.apply_gradients.assert_called_with(
                    zip.return_value, global_step=global_step)


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