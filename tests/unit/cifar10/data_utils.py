import unittest
from unittest.mock import patch

import numpy as np

from src.cifar10.data_utils import _read_data, read_data

class TestDataUtils(unittest.TestCase):
    @patch('src.cifar10.data_utils.pickle', return_value={ "data": np.random.rand(10, 32, 32, 3), "labels": np.ones((10)) })
    @patch('src.cifar10.data_utils.open')
    @patch('src.cifar10.data_utils.print')
    def test__read_data(self, print, open, pickle):
        _read_data("outputs", ["batch1"])
        print.assert_called_with("batch1")
        open.assert_called_with("outputs/batch1", 'rb')
        pickle.load.assert_called()

    @patch('src.cifar10.data_utils._read_data', return_value=(np.random.rand(45000, 32, 32, 3), np.ones((45000))))
    @patch('src.cifar10.data_utils.print')
    def test_read_data(self, print, _read_data):
        read_data("outputs")
        print.assert_any_call('-' * 80)
        print.assert_any_call("Reading data")
        _read_data.assert_any_call("outputs", [f"data_batch_{num + 1}" for num in range(5)])
        print.assert_any_call("Prepropcess: [subtract mean], [divide std]")