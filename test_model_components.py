import unittest

import torch

from model import MLP, Block, CausalSelfAttention, GPTConfig, LayerNorm


class TestLayerNorm(unittest.TestCase):
    def test_layer_norm_forward(self):
        # Initialize a LayerNorm module
        ndim = 256
        bias = True
        layer_norm = LayerNorm(ndim, bias)

        # Create input tensor
        input_tensor = torch.randn((32, ndim))

        # Perform a forward pass
        output = layer_norm(input_tensor)

        # Assertions based on expected output shapes or values
        self.assertEqual(output.shape, (32, ndim))
        self.assertTrue(torch.all(torch.isfinite(output)))


class TestCausalSelfAttention(unittest.TestCase):
    def test_causal_self_attention_forward(self):
        # Initialize a CausalSelfAttention module
        config = GPTConfig()
        causal_self_attention = CausalSelfAttention(config)

        # Create input tensor
        batch_size = 16
        seq_length = 64
        embedding_dim = config.n_embd
        input_tensor = torch.randn((batch_size, seq_length, embedding_dim))

        # Perform a forward pass
        output = causal_self_attention(input_tensor)

        # Assertions based on expected output shapes or values
        self.assertEqual(output.shape, (batch_size, seq_length, embedding_dim))
        self.assertTrue(torch.all(torch.isfinite(output)))


class TestMLP(unittest.TestCase):
    def test_mlp_forward(self):
        # Initialize an MLP module
        config = GPTConfig()
        mlp = MLP(config)

        # Create input tensor
        batch_size = 8
        seq_length = 32
        embedding_dim = config.n_embd
        input_tensor = torch.randn((batch_size, seq_length, embedding_dim))

        # Perform a forward pass
        output = mlp(input_tensor)

        # Assertions based on expected output shapes or values
        self.assertEqual(output.shape, (batch_size, seq_length, embedding_dim))
        self.assertTrue(torch.all(torch.isfinite(output)))


class TestBlock(unittest.TestCase):
    def test_block_forward(self):
        # Initialize a Block module
        config = GPTConfig()
        block = Block(config)

        # Create input tensor
        batch_size = 8
        seq_length = 32
        embedding_dim = config.n_embd
        input_tensor = torch.randn((batch_size, seq_length, embedding_dim))

        # Perform a forward pass
        output = block(input_tensor)

        # Assertions based on expected output shapes or values
        self.assertEqual(output.shape, (batch_size, seq_length, embedding_dim))
        self.assertTrue(torch.all(torch.isfinite(output)))


if __name__ == "__main__":
    unittest.main()
