#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn

"""Embedding is "向量映射", to transform some objects to vetor, or in other works, vectorize"""
class ModelEmbeddings(nn.Module):   # class inheritance from nn.module
    """
    Class that converts input words to their embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layers.

        @param embed_size (int): Embedding size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size

        # default values
        self.source = None
        self.target = None

        src_pad_token_idx = vocab.src['<pad>']
        tgt_pad_token_idx = vocab.tgt['<pad>']

        ### YOUR CODE HERE (~2 Lines)
        self.source = nn.Embedding(len(vocab.src), embed_size, padding_idx = src_pad_token_idx)
        self.target = nn.Embedding(len(vocab.tgt), embed_size, padding_idx = tgt_pad_token_idx)
        ### TODO - Initialize the following variables:
        ###     self.source (Embedding Layer for source language)
        ###     self.target (Embedding Layer for target langauge)
        ###
        ### Note:
        ###     1. `vocab` object contains two vocabularies:
        ###            `vocab.src` for source
        ###            `vocab.tgt` for target
        ###     2. You can get the length of a specific vocabulary by running:
        ###             `len(vocab.<specific_vocabulary>)`
        ###     3. Remember to include the padding token for the specific vocabulary
        ###        when creating your Embedding.
        ###
        ### Use the following docs to properly initialize these variables:
        ###     Embedding Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding

        ### END YOUR CODE
