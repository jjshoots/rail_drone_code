from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from wingman import NeuralBlocks


class SelfAttention(nn.Module):
    """A very simple self attention module with sinusoidal positional embeddings."""

    def __init__(self, embed_dim: int, num_heads: int, context_length: int):
        """A very simple self attention module with sinusoidal positional embeddings.

        The internal dimension of the model is embed_dim / num_heads

        Args:
            embed_dim (int): the dimension size of the embeddings
            num_heads (int): the number of heads in the model
            context_length (int): maximum context length allowable
        """
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), f"{embed_dim=} must be divisible by {num_heads=}."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)
        self.context_length = context_length
        self._normalization_const = np.sqrt(embed_dim)

        # positional encoding, split it into n heads, shape (num_heads, B, N, head_dim)
        encoding = SelfAttention.positional_encoding_1d(context_length, embed_dim).view(
            num_heads, 1, context_length, self.head_dim
        )
        self.pos_encoding: torch.Tensor
        self.register_buffer("pos_encoding", encoding)

        # qkv layers
        features = [self.head_dim] * 3
        activations = ["tanh"] * (len(features) - 1)
        self.q_nets = nn.ModuleList(
            [
                NeuralBlocks.generate_linear_stack(features, activations)
                for _ in range(num_heads)
            ]
        )
        self.k_nets = nn.ModuleList(
            [
                NeuralBlocks.generate_linear_stack(features, activations)
                for _ in range(num_heads)
            ]
        )
        self.v_nets = nn.ModuleList(
            [
                NeuralBlocks.generate_linear_stack(features, activations)
                for _ in range(num_heads)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expects inputs to be (batch_size, embedding_dim, *), where * is any number of dimensions.

        Args:
            x (int): An (batch_size, embedding_dim, *) shaped tensor, where * is any number of dimensions

        Returns:
            torch.Tensor: A (batch_size, embedding_dim, *) tensor
        """
        assert (
            len(x.shape) >= 2
        ), f"Input must be shape (batch_size, embedding_dim, *) where * is any number of dimensions, got {x.shape}."
        assert (
            x.shape[1] == self.embed_dim
        ), f"The size of vector {x.shape[1]=} must equal {self.embed_dim=}."
        assert (
            np.prod(x.shape[2:]) <= self.context_length
        ), f"tensor free dimension must be larger than {self.context_length=}, got {np.prod(x.shape[2:])=}."

        # store the shapes for reconstruction later
        shape_BE = x.shape[:2]
        shape_other = x.shape[2:]

        # convert tensor to be (B, num_heads, head_dim, N)
        x = x.view(shape_BE[0], self.num_heads, self.head_dim, -1)

        # convert tensor to be (num_heads, B, N, head_dim)
        x = x.permute((1, 0, 3, 2))

        # add positional encoding
        if self.training:
            starts = np.random.randint(
                self.context_length - x.shape[-2], size=(x.shape[-3],)
            )
            ends = starts + x.shape[-2]
            encoding = torch.cat(
                [self.pos_encoding[..., s:e, :] for s, e in zip(starts, ends)], dim=1
            )
            x = x + encoding
        else:
            x = x + self.pos_encoding[..., : x.shape[-2], :]

        # get key, queries, values
        qs = torch.stack([q_net(s) for q_net, s in zip(self.q_nets, x)])
        ks = torch.stack([k_net(s) for k_net, s in zip(self.k_nets, x)])
        vs = torch.stack([v_net(s) for v_net, s in zip(self.v_nets, x)])

        # perform qkv computation, shape is (num_heads, B, N, head_dim)
        scores = torch.matmul(qs, ks.swapaxes(-1, -2)) / self._normalization_const
        scores = torch.softmax(scores, dim=-1)

        # attention and add
        y = torch.matmul(scores, vs) + x

        # convert tensor to be (B, N, num_heads, head_dim)
        y = y.permute((1, 2, 0, 3))

        # convert tensor to be (B, *, E), then normalize
        y = y.reshape(shape_BE[0], *shape_other, self.embed_dim)
        y = self.layer_norm(y)

        # reconstruct output to be (B, E, *)
        y = torch.moveaxis(y, -1, 1)

        return y

    @classmethod
    def positional_encoding_1d(cls, length: int, embed_dim: int) -> torch.Tensor:
        """positional_encoding_1d.

        Args:
            length (int): context length of the positional encoding
            embed_dim (int): the dimension size of the embeddings

        Returns:
            torch.Tensor: a (length, embed_dim) sinusoidal positional encoding
        """
        if embed_dim % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(embed_dim)
            )
        pe = torch.zeros(length, embed_dim)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * -(np.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe


# for testing lol
if __name__ == "__main__":
    att = SelfAttention(256, 8, 10000)
    x = torch.zeros(8, 256, 2, 4, 5, 3, 4)
