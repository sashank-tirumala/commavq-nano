import unittest

import torch

from model import GPT, GPTConfig  # Import necessary modules


class TestGPT(unittest.TestCase):
    def setUp(self):
        # Initialize a sample GPT configuration for testing
        self.config = GPTConfig(
            vocab_size=1000,
            n_embd=256,
            n_head=8,
            n_layer=6,
            dropout=0.1,
            bias=True,
            block_size=128,
        )

        # Create a sample input tensor for testing
        self.sample_input = torch.randint(0, self.config.vocab_size, (4, 16))

    def test_forward_pass(self):
        # Test the forward pass of the GPT model
        model = GPT(self.config)
        logits, loss = model(self.sample_input)

        # Check that the output shapes are as expected
        self.assertEqual(logits.shape, (4, 1, self.config.vocab_size))
        self.assertIsNone(loss)

    def test_forward_pass_with_targets(self):
        # Test the forward pass with target labels
        target_labels = torch.randint(0, self.config.vocab_size, (4, 16))

        model = GPT(self.config)
        logits, loss = model(self.sample_input, target_labels)

        # Check that the output shapes are as expected
        self.assertEqual(logits.shape, (4, 16, self.config.vocab_size))
        self.assertIsNotNone(loss)

    ## debug this test case
    # def test_configure_optimizers(self):
    #     # Test the configure_optimizers method
    #     device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     weight_decay = torch.tensor(0.01, dtype=torch.float32).to(device=device_type)
    #     learning_rate = torch.tensor(1e-3, dtype=torch.float32).to(device=device_type)
    #     betas = (torch.tensor(0.9, dtype=torch.float32).to(device=device_type), torch.tensor(0.95, dtype=torch.float32).to(device=device_type))

    #     model = GPT(self.config)
    #     optimizer = model.configure_optimizers(weight_decay, learning_rate, betas, device_type)

    #     # Check that the optimizer is an instance of AdamW
    #     self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_crop_block_size(self):
        # Test the crop_block_size method
        new_block_size = 64  # Smaller block size
        model = GPT(self.config)

        # Ensure that the initial block size matches the configuration
        self.assertEqual(model.config.block_size, self.config.block_size)

        # Crop the block size
        model.crop_block_size(new_block_size)

        # Check that the block size is updated
        self.assertEqual(model.config.block_size, new_block_size)

    def test_edge_case_large_block_size(self):
        # Test an edge case with a large block size
        large_block_size = 1024
        model = GPT(self.config)

        # Ensure that the initial block size matches the configuration
        self.assertEqual(model.config.block_size, self.config.block_size)

        # Crop the block size to a large value
        with self.assertRaises(AssertionError):
            model.crop_block_size(large_block_size)

    def test_edge_case_no_targets(self):
        # Test an edge case where no target labels are provided
        model = GPT(self.config)
        logits, loss = model(self.sample_input)

        # Check that the output shapes are as expected
        self.assertEqual(logits.shape, (4, 1, self.config.vocab_size))
        self.assertIsNone(loss)


if __name__ == "__main__":
    unittest.main()
