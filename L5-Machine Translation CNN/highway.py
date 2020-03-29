#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn




### YOUR CODE HERE for part 1d
class Highway(nn.Module):
    """ Class that map x_conv to an embedding vector """

    def __init__(self, word_embed_size):
        """
        Init the Highway module
        @param word_embed_size (int): Embedding size (dimensionality) for both the input (conv_out) and output
        """
        super(Highway, self).__init__()
        self.word_embed_size = word_embed_size
        self.proj = nn.Linear(self.word_embed_size, self.word_embed_size)
        self.gate = nn.Linear(self.word_embed_size, self.word_embed_size)

    def forward(self, x_conv_out):
        """
        Take a mini batch of convonlution output, compute
        @param x_conv_out (Torch.tensor): Tensor of x_conv_out of shape (batch_size * word_embed_size)
        @return:Tensor of word_embedding of shape (batch_size * word_embed_size)
        """
        x_proj = torch.relu(self.proj(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))

        x = x_gate * x_proj + (torch.ones(self.word_embed_size) - x_gate) * x_conv_out
        return x
### END YOUR CODE





































































