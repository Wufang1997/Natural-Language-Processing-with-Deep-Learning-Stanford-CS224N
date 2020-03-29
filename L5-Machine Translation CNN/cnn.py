#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


### YOUR CODE HERE for part 1e
class CNN(nn.Module):
    def __init__(self, char_embed_size, word_length, num_filters, kernel_size=5):
        """
        Init the character CNN module
        @param char_embed_size (int):embedding size of each character in a word
        @param word_length (int): Max word length
        @param num_filters (int): Number of filters (or kernel)
        @param kernel_size (int): Size of kernels
        """
        super(CNN,self).__init__()
        self.char_embed_size = char_embed_size
        self.word_length = word_length
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.Conv1d = nn.Conv1d(
            in_channels=self.char_embed_size,
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            bias=True
        )

        self.max_pool_1d = nn.MaxPool1d(kernel_size=self.word_length - self.kernel_size + 1,
                                        stride=None, padding=0, dilation=1)

    def forward(self, x_reshape):
        """
        Take a mini batch of character embedding of each word, compute word embedding
        @param x_reshape (Tensor): A dense reshaped character embedding of shape (batch_size * char_embed_size * word_length)
        @return (Tensor): Raw word embedding of each word in batch
        """
        x_conv = self.Conv1d(x_reshape)  # Shape (batch_size * num_filters * (word_length - kernek_size + 1))
        x_conv_out = self.max_pool_1d(F.relu(x_conv))   # Shape (batch_size * num_filters * 1)
        x_conv_out = torch.squeeze(x_conv_out)   # Shape (batch_size * num_filters)

        return x_conv_out

### END YOUR CODE























































