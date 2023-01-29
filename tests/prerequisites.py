import unittest
import os
import tensorflow as tf

class TestPrerequisites(unittest.TestCase):
    def test_sufficient_physical_memory(self):
        self.assertGreater(
            os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'),
            31 * 1024 * 1024 * 1024,
            "Prerequisite test failed due to insufficient physical memory.")

    def test_intel_tensorflow(self):
        from tensorflow.python.util import _pywrap_util_port
        self.assertTrue(
            _pywrap_util_port.IsMklEnabled(),
            "Prerequisite test failed because Intel TensorFlow not detected.")

if __name__ == "__main__":
    unittest.main()
