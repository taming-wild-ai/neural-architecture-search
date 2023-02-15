import unittest
from unittest.mock import patch

from src.common_ops import lstm, stack_lstm, create_weight

class TestCommonOps(unittest.TestCase):

    @patch('src.common_ops.tf.tanh', return_value=0.5)
    @patch('src.common_ops.tf.sigmoid', return_value=0.5)
    @patch('src.common_ops.tf.split', return_value=(0.0, 0.0, 0.0, 0.0))
    @patch('src.common_ops.tf.matmul', return_value="matmul")
    @patch('src.common_ops.tf.concat', return_value="concat")
    def test_lstm(self, concat, matmul, split, sigmoid, tanh):
        self.assertEqual((1.25, 0.25), lstm(1, 2, 3, 4))
        concat.assert_called_with([1, 3], axis=1)
        matmul.assert_called_with("concat", 4)
        split.assert_called_with("matmul", 4, axis=1)

    @patch('src.common_ops.lstm', return_value=(1.25, 0.25))
    def test_stack_lstm(self, lstm):
        self.assertEqual(([1.25], [0.25]), stack_lstm(1, [2], [3], [4]))
        lstm.assert_called_with(1, 2, 3, 4)

    @patch('src.common_ops.tf.compat.v1.get_variable', return_value="get_variable")
    @patch('src.common_ops.tf.keras.initializers.he_normal', return_value="he_normal")
    def test_create_weight(self, hn, get_variable):
        self.assertEqual("get_variable", create_weight(None, None))

if "__main__" == "__name__":
    unittest.main()
