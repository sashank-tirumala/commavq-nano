import pdb
import unittest

import numpy as np
import torch

from model import CausalSelfAttention, GPTConfig  # Import necessary modules


class TestCausalSelfAttention(unittest.TestCase):
    def setUp(self):
        # Initialize a sample GPT configuration for testing
        self.config = GPTConfig(
            n_embd=256, n_head=8, dropout=0.1, bias=False, block_size=128
        )

    def test_forward_pass(self):
        # Test the forward pass of CausalSelfAttention
        batch_size = 4
        sequence_length = 16
        embedding_dim = self.config.n_embd
        inputs = torch.randn(batch_size, sequence_length, embedding_dim)

        attention = CausalSelfAttention(self.config)
        outputs = attention(inputs)

        # Check if the output shape matches the input shape
        self.assertEqual(outputs.shape, (batch_size, sequence_length, embedding_dim))

    def test_no_bias(self):
        # Test behavior when bias is set to False
        config_no_bias = GPTConfig(
            n_embd=256, n_head=8, dropout=0.1, bias=False, block_size=128
        )
        batch_size = 4
        sequence_length = 16
        embedding_dim = config_no_bias.n_embd
        inputs = torch.randn(batch_size, sequence_length, embedding_dim)

        attention = CausalSelfAttention(config_no_bias)
        outputs = attention(inputs)

        # Check that the bias parameter is None
        flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not flash:
            self.assertIsNone(attention.bias)
        else:
            self.assertIsNone(attention.c_attn.bias)

    def test_dropout(self):
        # Test the dropout behavior
        config_dropout = GPTConfig(
            n_embd=256, n_head=8, dropout=0.2, bias=True, block_size=128
        )
        batch_size = 4
        sequence_length = 16
        embedding_dim = config_dropout.n_embd
        inputs = torch.randn(batch_size, sequence_length, embedding_dim)

        attention = CausalSelfAttention(config_dropout)
        outputs = attention(inputs)

        # Check that dropout is applied during training
        self.assertTrue(attention.training)

    # def test_causal_mask(self):
    #     # Test that the causal mask is correctly applied, have to split the computation in forward to get the output.
    #     batch_size = 4
    #     sequence_length = 16
    #     embedding_dim = self.config.n_embd
    #     inputs = torch.randn(batch_size, sequence_length, embedding_dim)

    #     attention = CausalSelfAttention(self.config)
    #     outputs = attention(inputs)

    #     # Check that the attention values are zero in the upper triangle of the attention matrix
    #     flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    #     if not flash:
    #         causal_mask = attention.bias[0, 0, :, :].cpu().numpy()
    #         upper_triangle = causal_mask[np.triu_indices(sequence_length, k=1)]
    #         self.assertTrue((upper_triangle == -float('inf')).all())
    #     pdb.set_trace()
    #     self.assertTrue((outputs[0, :, -1, :sequence_length - 1] == 0).all())


if __name__ == "__main__":
    unittest.main()
