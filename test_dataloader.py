import os
import unittest

import numpy as np
import torch

from train import \
    get_batch  # Replace 'your_module' with the actual module name


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Create temporary data for testing
        self.data_dir = "temp_data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.train_data = np.arange(1000, dtype=np.uint16)
        self.val_data = np.arange(1000, 2000, dtype=np.uint16)
        self.block_size = 32
        self.batch_size = 8

        np.save(os.path.join(self.data_dir, "train.bin"), self.train_data)
        np.save(os.path.join(self.data_dir, "val.bin"), self.val_data)

    def tearDown(self):
        # Clean up temporary data directory
        os.remove(os.path.join(self.data_dir, "train.bin.npy"))
        os.remove(os.path.join(self.data_dir, "val.bin.npy"))
        os.rmdir(self.data_dir)

    def test_get_batch_train(self):
        split = "train"
        x, y = get_batch(split, self.data_dir, self.block_size, self.batch_size)

        # Check if x and y have the correct shapes
        self.assertEqual(x.shape, (self.batch_size, self.block_size))
        self.assertEqual(y.shape, (self.batch_size, self.block_size))

        # Check if x and y contain values from the 'train_data'
        expected_x = torch.from_numpy(self.train_data[: self.block_size])
        expected_y = torch.from_numpy(self.train_data[1 : self.block_size + 1])

        self.assertTrue(torch.equal(x[0], expected_x))
        self.assertTrue(torch.equal(y[0], expected_y))

    def test_get_batch_val(self):
        split = "val"
        x, y = get_batch(split, self.data_dir, self.block_size, self.batch_size)

        # Check if x and y have the correct shapes
        self.assertEqual(x.shape, (self.batch_size, self.block_size))
        self.assertEqual(y.shape, (self.batch_size, self.block_size))

        # Check if x and y contain values from the 'val_data'
        expected_x = torch.from_numpy(self.val_data[: self.block_size])
        expected_y = torch.from_numpy(self.val_data[1 : self.block_size + 1])

        self.assertTrue(torch.equal(x[0], expected_x))
        self.assertTrue(torch.equal(y[0], expected_y))


if __name__ == "__main__":
    unittest.main()
