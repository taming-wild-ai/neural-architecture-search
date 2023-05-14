import unittest
import sys
import unittest.mock as mock
from unittest.mock import patch

import src.cifar10.main as main


class RestoreFLAGS:
    main.fw.FLAGS(sys.argv) # Need to parse flags before accessing them

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.old_values = {}

    def __enter__(self):
        for key, value in self.kwargs.items():
            self.old_values[key] = main.fw.FLAGS.__getattr__(key)
            main.fw.FLAGS.__setattr__(key, value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self.old_values.items():
            main.fw.FLAGS.__setattr__(key, value)


class TestCIFAR10Main(unittest.TestCase):
    @patch('src.cifar10.main.os.path.isdir', return_value=False)
    @patch('src.cifar10.main.Logger', return_value=sys.stdout)
    @patch('src.cifar10.main.train')
    @patch('src.cifar10.main.print')
    @patch('src.cifar10.main.shutil.rmtree')
    @patch('src.cifar10.main.os.makedirs')
    @patch('src.cifar10.main.print_user_flags')
    def test_main_creates_output_directory(self, print_flags, makedirs, rmtree, print, train, _logger, _isdir):
        main.main(None)

        print.assert_any_call(("-" * 80))
        print.assert_any_call("Path  does not exist. Creating.")
        rmtree.assert_not_called()
        makedirs.assert_called_with("")
        print_flags.assert_called_with()
        train.assert_called_with()

    @patch('src.cifar10.main.os.path.isdir', return_value=True)
    @patch('src.cifar10.main.Logger', return_value=sys.stdout)
    @patch('src.cifar10.main.print_user_flags')
    @patch('src.cifar10.main.os.makedirs')
    @patch('src.cifar10.main.shutil.rmtree')
    @patch('src.cifar10.main.train')
    @patch('src.cifar10.main.print')
    def test_main_reset_output_dir(self, print, train, rmtree, makedirs, print_flags, _logger, _isdir):
        with RestoreFLAGS(reset_output_dir=True):
            main.main(None)

            print.assert_any_call(("-" * 80))
            print.assert_any_call("Path  exists. Remove and remake.")
            rmtree.assert_called_with("")
            makedirs.assert_called_with('')
            print_flags.assert_called_with()
            train.assert_called_with()

    @patch('src.cifar10.main.os.path.isdir', return_value=True)
    @patch('src.cifar10.main.Logger', return_value=sys.stdout)
    @patch('src.cifar10.main.print_user_flags')
    @patch('src.cifar10.main.os.makedirs')
    @patch('src.cifar10.main.shutil.rmtree')
    @patch('src.cifar10.main.train')
    @patch('src.cifar10.main.print')
    def test_main_ignore_output_dir(self, print, train, rmtree, makedirs, print_flags, _logger, _isdir):
        with RestoreFLAGS(reset_output_dir=False):
            main.main(None)

            print.assert_any_call(("-" * 80))
            rmtree.assert_not_called()
            makedirs.assert_not_called()
            print_flags.assert_called_with()
            train.assert_called_with()

    def test_get_ops_fails_assertion(self):
        with RestoreFLAGS(search_for=None):
            self.assertRaises(AssertionError, main.get_ops, None, None)

    @patch('src.cifar10.main.MicroController')
    @patch('src.cifar10.main.MicroChild')
    @patch('src.cifar10.main.MacroController')
    @patch('src.cifar10.main.MacroChild')
    def test_get_ops_micro_fails_assertion(self, _gch_ctor, _gco_ctor, _mch_ctor, _mco_ctor):
        with RestoreFLAGS(search_for="micro", child_fixed_arc="", controller_training=True):
            self.assertRaises(AssertionError, main.get_ops, None, None)

    @patch('src.cifar10.main.MicroController')
    @patch('src.cifar10.main.MicroChild')
    @patch('src.cifar10.main.MacroController')
    @patch('src.cifar10.main.MacroChild')
    def test_get_ops_micro(self, gch_ctor, gco_ctor, mch_ctor, mco_ctor):
        mco = mock.MagicMock()
        mco.build_trainer = mock.MagicMock(return_value=('train_op', 'lr', 'grad_norm', 'optimizer'))
        mco_ctor.return_value = mco
        mch = mock.MagicMock()
        mch.connect_controller = mock.MagicMock(return_value=('train_op', 'lr', 'grad_norm', 'optimizer'))
        mch.ValidationRLShuffle = mock.MagicMock(return_value=mock.MagicMock(return_value=('x_valid_shuffle', 'y_valid_shuffle')))
        mch_ctor.return_value = mch
        with RestoreFLAGS(search_for="micro", controller_training=True, child_fixed_arc=None):
            main.get_ops(None, None)

        mco_ctor.assert_called_with(lstm_size=64, lstm_num_layers=1, lstm_keep_prob=1.0, lr_dec_start=0, lr_dec_every=1000000, optim_algo='adam')
        mch_ctor.assert_called_with(None, None, clip_mode='norm', optim_algo='momentum')
        gco_ctor.assert_not_called()
        gch_ctor.assert_not_called()
        mco.build_trainer.assert_called_with(mch, mch.ValidationRL())
        mch.connect_controller.assert_called_with(mco)

    @patch('src.cifar10.main.MicroController')
    @patch('src.cifar10.main.MicroChild')
    @patch('src.cifar10.main.MacroController')
    @patch('src.cifar10.main.MacroChild')
    def test_get_ops_macro(self, gch_ctor, gco_ctor, mch_ctor, mco_ctor):
        gco = mock.MagicMock()
        gco.build_trainer = mock.MagicMock(return_value=('train_op', 'lr', 'grad_norm', 'optimizer'))
        gco_ctor.return_value = gco
        gch = mock.MagicMock()
        gch.connect_controller = mock.MagicMock(return_value=('train_op', 'lr', 'grad_norm', 'optimizer'))
        gch.ValidationRLShuffle = mock.MagicMock(return_value=mock.MagicMock(return_value=('x_valid_shuffle', 'y_valid_shuffle')))
        gch_ctor.return_value = gch
        with RestoreFLAGS(search_for="macro", controller_training=True, child_fixed_arc=None):
            main.get_ops(None, None)

        mco_ctor.assert_not_called()
        mch_ctor.assert_not_called()
        gco_ctor.assert_called_with(lstm_size=64, lstm_num_layers=1, lstm_keep_prob=1.0, lr_dec_start=0, lr_dec_every=1000000, optim_algo='adam')
        gch_ctor.assert_called_with(None, None, clip_mode='norm', optim_algo='momentum')
        gch.connect_controller.assert_called_with(gco)
        gco.build_trainer.assert_called_with(gch, gch.ValidationRL())

    @patch('src.cifar10.main.MicroController')
    @patch('src.cifar10.main.MicroChild')
    @patch('src.cifar10.main.MacroController')
    @patch('src.cifar10.main.MacroChild')
    def test_get_ops_macro_child_only(self, gch_ctor, gco_ctor, mch_ctor, mco_ctor):
        gch = mock.MagicMock()
        gch.connect_controller = mock.MagicMock(return_value=('train_op', 'lr', 'grad_norm', 'optimizer'))
        gch_ctor.return_value = gch
        with RestoreFLAGS(search_for="macro", controller_training=False, child_fixed_arc=""):
            main.get_ops(None, None)

        mco_ctor.assert_not_called()
        mch_ctor.assert_not_called()
        gco_ctor.assert_not_called()
        gch_ctor.assert_called_with(None, None, clip_mode='norm', optim_algo='momentum')
        gch.connect_controller.assert_called_with(None)

    @staticmethod
    def mock_session_run(ops):
        if list == type(ops):
            if 2 == len(ops):
                return (["0", "1"], 0.95)
            elif 5 == len(ops):
                return (1, 2, 3, 4, None)
            elif 6 == len(ops):
                return (1, 2, 3, 4, 5, None)
            elif 8 == len(ops):
                return (1, 2, 3, 4, 5, 6, 7, None)
        else:
            return 311 # global_step

    @staticmethod
    def mock_get_ops(controller_optimizer, child_optimizer, eval_func):
        return mock.MagicMock(
            return_value={
                "eval_func": eval_func,
                "eval_every": 1,
                "num_train_batches": 1,
                "child": {
                    'model': mock.MagicMock(name='child model', return_value='logits'),
                    'x_valid_shuffle': 'x_valid_shuffle',
                    'y_valid_shuffle': 'y_valid_shuffle',
                    'images': mock.MagicMock(name='images'),
                    "num_train_batches": 1,
                    "loss": mock.MagicMock(return_value=2.0),
                    'train_loss': mock.MagicMock(return_value=2.0),
                    "lr": 0.01,
                    "grad_norm": mock.MagicMock(return_value=5),
                    "train_acc": mock.MagicMock(return_value=0.01),
                    "train_op": mock.MagicMock(return_value={}),
                    "global_step": 311,
                    "optimizer": child_optimizer},
                "controller": {
                    'model': mock.MagicMock(name='controller model'),
                    "optimizer": controller_optimizer,
                    "loss": mock.MagicMock(return_value=2.0),
                    "entropy": 2.0,
                    "lr": 0.01,
                    "grad_norm": mock.MagicMock(return_value=0.01),
                    "valid_acc": mock.MagicMock(return_value=0.25),
                    "baseline": 2.0,
                    "skip_rate": 0.1,
                    "train_op": mock.MagicMock(return_value=2.0),
                    "train_step": 311,
                    "sample_arc": "0" }})

    @patch('src.cifar10.main.read_data', return_value=(None, None))
    @patch('src.cifar10.main.fw.Saver')
    @patch('src.cifar10.main.fw.ConfigProto')
    @patch('src.cifar10.main.fw.Hook')
    @patch('src.cifar10.main.print')
    def test_train(self, print, hook, cp, saver, _rd):
        with RestoreFLAGS(
            child_sync_replicas=True,
            child_num_aggregate=1,
            controller_training=True,
            controller_sync_replicas=True,
            child_fixed_arc=None):
            child_optimizer = mock.MagicMock()
            controller_optimizer = mock.MagicMock()
            main.get_ops = self.mock_get_ops(controller_optimizer, child_optimizer, mock.MagicMock())
            mock_session = mock.MagicMock()
            mock_session.run = self.mock_session_run
            mock_session_context_mgr = mock.MagicMock()
            mock_session_context_mgr.__enter__ = mock.MagicMock(return_value=mock_session)
            with patch('src.cifar10.main.fw.Session', return_value=mock_session_context_mgr) as sess:
                main.train()
                print.assert_any_call(("-" * 80))
                print.assert_any_call("Starting session")
                cp.assert_called_with()
                sess.assert_called_with(config=cp(), hooks=[hook(), child_optimizer.make_session_run_hook(True), controller_optimizer.make_session_run_hook(True)], checkpoint_dir='')
                child_optimizer.make_session_run_hook.assert_called_with(True)
                saver.assert_called_with()
                controller_optimizer.make_session_run_hook.assert_called_with(True)

    @patch('src.cifar10.main.read_data', return_value=(None, None))
    @patch('src.cifar10.main.fw.Saver')
    @patch('src.cifar10.main.fw.ConfigProto')
    @patch('src.cifar10.main.fw.Hook')
    @patch('src.cifar10.main.print')
    def test_train_child_only1(self, print, hook, cp, saver, _rd):
        with RestoreFLAGS(
            child_sync_replicas=False,
            controller_sync_replicas=False,
            controller_training=False,
            child_fixed_arc="",
            log_every=1):
            child_optimizer = mock.MagicMock()
            controller_optimizer = mock.MagicMock()
            eval_func = mock.MagicMock()
            main.get_ops = self.mock_get_ops(controller_optimizer, child_optimizer, eval_func)
            mock_session = mock.MagicMock()
            mock_session.run = self.mock_session_run
            mock_session_context_mgr = mock.MagicMock()
            mock_session_context_mgr.__enter__ = mock.MagicMock(return_value=mock_session)
            with patch('src.cifar10.main.fw.Session', return_value=mock_session_context_mgr) as sess:
                main.train()
                saver.assert_any_call()
                child_optimizer.make_session_run_hook.assert_not_called()
                controller_optimizer.make_session_run_hook.assert_not_called()
                print.assert_any_call(("-" * 80))
                print.assert_any_call("Starting session")
                sess.assert_called_with(config=cp(), hooks=[hook()], checkpoint_dir='')
                print.assert_called_with("Epoch 311: Eval")
                eval_func.assert_called_with(mock_session, "test")

    @patch('src.cifar10.main.read_data', return_value=(None, None))
    @patch('src.cifar10.main.fw.Saver')
    @patch('src.cifar10.main.fw.ConfigProto')
    @patch('src.cifar10.main.fw.Hook')
    @patch('src.cifar10.main.print')
    def test_train_child_only2(self, print, hook, cp, saver, _rd):
        with RestoreFLAGS(
            child_sync_replicas=False,
            controller_sync_replicas=True,
            controller_training=True,
            child_fixed_arc="",
            search_for="micro",
            controller_train_every=1):
            child_optimizer = mock.MagicMock()
            controller_optimizer = mock.MagicMock()
            eval_func = mock.MagicMock()
            main.get_ops = self.mock_get_ops(controller_optimizer, child_optimizer, eval_func)
            mock_session = mock.MagicMock()
            mock_session.run = self.mock_session_run # mock.MagicMock(return_value=(1, 2, 3, 4, None))
            mock_session_context_mgr = mock.MagicMock()
            mock_session_context_mgr.__enter__ = mock.MagicMock(return_value=mock_session)
            with patch('src.cifar10.main.fw.Session', return_value=mock_session_context_mgr) as sess:
                main.train()
                saver.assert_any_call()
                controller_optimizer.make_session_run_hooks.assert_not_called()
                print.assert_any_call(("-" * 80))
                print.assert_any_call("Starting session")
                sess.assert_called_with(config=cp(), hooks=[hook(), controller_optimizer.make_session_run_hook(True)], checkpoint_dir='')
                print.assert_any_call("Epoch 311: Training controller")
                print.assert_called_with("Epoch 311: Eval")
                eval_func.assert_called_with(mock_session, "test")

    @patch('src.cifar10.main.read_data', return_value=(None, None))
    @patch('src.cifar10.main.fw.Saver')
    @patch('src.cifar10.main.fw.ConfigProto')
    @patch('src.cifar10.main.fw.Hook')
    @patch('src.cifar10.main.print')
    def test_train_child_only3(self, print, hook, cp, saver, _rd):
        with RestoreFLAGS(
            child_sync_replicas=False,
            controller_sync_replicas=True,
            controller_training=True,
            child_fixed_arc=None,
            search_for="macro",
            controller_train_every=1):
            child_optimizer = mock.MagicMock()
            controller_optimizer = mock.MagicMock()
            eval_func = mock.MagicMock()
            main.get_ops = self.mock_get_ops(controller_optimizer, child_optimizer, eval_func)
            mock_session = mock.MagicMock()
            mock_session.run = self.mock_session_run
            mock_session_context_mgr = mock.MagicMock()
            mock_session_context_mgr.__enter__ = mock.MagicMock(return_value=mock_session)
            with patch('src.cifar10.main.fw.Session', return_value=mock_session_context_mgr) as sess:
                main.train()

                saver.assert_any_call()
                controller_optimizer.make_session_run_hook.assert_called_with(True)
                print.assert_any_call("-" * 80)
                print.assert_any_call("Starting session")
                sess.assert_called_with(config=cp(), hooks=[hook(), controller_optimizer.make_session_run_hook(True)], checkpoint_dir='')
                print.assert_any_call("Epoch 311: Training controller")
                print.assert_called_with("Epoch 311: Eval")
                eval_func.assert_called_with(mock_session, "test")

    @patch('src.cifar10.main.read_data', return_value=(None, None))
    @patch('src.cifar10.main.fw.Saver')
    @patch('src.cifar10.main.print')
    def test_train_child_only_macro(self, print, saver, _rd):
        with RestoreFLAGS(
            child_sync_replicas=False,
            controller_sync_replicas=True,
            controller_training=True,
            child_fixed_arc=None,
            search_for=None,
            controller_train_every=1):
            child_optimizer = mock.MagicMock()
            controller_optimizer = mock.MagicMock()
            main.get_ops = self.mock_get_ops(controller_optimizer, child_optimizer, mock.MagicMock())
            mock_session = mock.MagicMock()
            mock_session.run = self.mock_session_run
            mock_session_context_mgr = mock.MagicMock()
            mock_session_context_mgr.__enter__ = mock.MagicMock(return_value=mock_session)
            with patch('src.cifar10.main.fw.Session', return_value=mock_session_context_mgr):

                main.train()

                controller_optimizer.make_session_run_hook.assert_called_with(True)
                print.assert_any_call(("-" * 80))
                print.assert_any_call("Starting session")
                saver.assert_any_call()

    @patch('src.cifar10.main.read_data', return_value=(None, None))
    @patch('src.cifar10.main.fw.Saver')
    @patch('src.cifar10.main.fw.ConfigProto')
    @patch('src.cifar10.main.fw.Hook')
    @patch('src.cifar10.main.print')
    def test_train_child_only_macro_whole_channels(self, print, hook, cp, saver, _rd):
        with RestoreFLAGS(
            child_sync_replicas=False,
            controller_sync_replicas=True,
            controller_training=True,
            child_fixed_arc=None,
            search_for=None,
            controller_search_whole_channels=True,
            controller_train_every=1):
            child_optimizer = mock.MagicMock()
            controller_optimizer = mock.MagicMock()
            main.get_ops = self.mock_get_ops(controller_optimizer, child_optimizer, mock.MagicMock())
            mock_session = mock.MagicMock()
            mock_session.run = self.mock_session_run
            mock_session_context_mgr = mock.MagicMock()
            mock_session_context_mgr.__enter__ = mock.MagicMock(return_value=mock_session)
            with patch('src.cifar10.main.fw.Session', return_value=mock_session_context_mgr) as sess:
                main.train()
                controller_optimizer.make_session_run_hooks.assert_not_called()
                print.assert_any_call(("-" * 80))
                print.assert_any_call("Starting session")
                saver.assert_any_call()
                sess.assert_called_with(config=cp(), hooks=[hook(), controller_optimizer.make_session_run_hook(True)], checkpoint_dir='')