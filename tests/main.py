import unittest
import sys
import unittest.mock as mock

import src.cifar10.main as main


class TestCIFAR10Main(unittest.TestCase):
    def test_main_creates_output_directory(self):
        main.print = mock.MagicMock()
        main.os.path.isdir = mock.MagicMock(return_value = False)
        main.shutil.rmtree = mock.MagicMock()
        main.os.makedirs = mock.MagicMock()
        main.Logger = mock.MagicMock(return_value=sys.stdout)
        main.utils.print_user_flags = mock.MagicMock()
        main.train = mock.MagicMock()

        main.main(None)

        main.print.assert_any_call(("-" * 80))
        main.print.assert_any_call("Path  does not exist. Creating.")
        main.shutil.rmtree.assert_not_called()
        main.os.makedirs.assert_called_with("")
        main.utils.print_user_flags.assert_called_with()
        main.train.assert_called_with()

    def test_main_reset_output_dir(self):
        main.print = mock.MagicMock()
        main.os.path.isdir = mock.MagicMock(return_value=True)
        main.FLAGS.reset_output_dir = True
        main.shutil.rmtree = mock.MagicMock()
        main.os.makedirs = mock.MagicMock()
        main.Logger = mock.MagicMock(return_value=sys.stdout)
        main.utils.print_user_flags = mock.MagicMock()
        main.train = mock.MagicMock()

        main.main(None)

        main.print.assert_any_call(("-" * 80))
        main.print.assert_any_call("Path  exists. Remove and remake.")
        main.shutil.rmtree.assert_called_with("")
        main.utils.print_user_flags.assert_called_with()
        main.train.assert_called_with()

    def test_main_ignore_output_dir(self):
        main.print = mock.MagicMock()
        main.os.path.isdir = mock.MagicMock(return_value=True)
        main.FLAGS.reset_output_dir = False
        main.shutil.rmtree = mock.MagicMock()
        main.os.makedirs = mock.MagicMock()
        main.Logger = mock.MagicMock(return_value=sys.stdout)
        main.utils.print_user_flags = mock.MagicMock()
        main.train = mock.MagicMock()

        main.main(None)

        main.print.assert_any_call(("-" * 80))
        main.shutil.rmtree.assert_not_called()
        main.utils.print_user_flags.assert_called_with()
        main.train.assert_called_with()
