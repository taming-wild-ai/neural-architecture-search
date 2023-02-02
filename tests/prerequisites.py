import unittest
import os
import tensorflow as tf

class TestPrerequisites(unittest.TestCase):
    def test_sufficient_physical_memory(self):
        self.assertGreater(
            os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'),
            31 * 1024 * 1024 * 1024,
            "Prerequisite test failed due to insufficient physical memory.")

    def test_gpu_presence(self):
        detected = False
        for device in tf.config.list_physical_devices():
            detected = detected or device.device_type == 'GPU'
        self.assertTrue(detected), "Prerequisite test failed because GPU not detected."

if __name__ == "__main__":
    unittest.main()
