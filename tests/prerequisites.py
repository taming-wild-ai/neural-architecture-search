import unittest
import os
import tensorflow as tf

class TestStringMethods(unittest.TestCase):

    def test_sufficient_physical_memory(self):
        self.assertGreater(
            os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'),
            31 * 1024 * 1024 * 1024)

    def test_gpu_or_intel_tensorflow(self):
        def intel_tensorflow_installed():
            from tensorflow.python.util import _pywrap_util_port
            return _pywrap_util_port.IsMklEnabled()
        self.assertTrue(tf.test.is_gpu_available() or intel_tensorflow_installed())

if __name__ == '__main__':
    unittest.main()
