#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        self.embed_size = embed_size
        """
        there, the embed_size is the number of the filters, instead of the size of the character embeddings.
        """
        self.vocab = vocab
        self.char_embed_size = 50
        self.word_length = 21
        self.dropout_rate = 0.3

        self.char_embeddings = nn.Embedding(len(self.vocab.char2id), self.char_embed_size,
                                       padding_idx=vocab.char2id['<pad>'])
        """
        A simple lookup table that stores embeddings of a fixed dictionary and size.
        This module is often used to store word embeddings and retrieve them using indices
        """
        self.CNN = CNN(char_embed_size=self.char_embed_size, num_filters=self.embed_size, word_length=self.word_length)
        self.Highway = Highway(word_embed_size=self.embed_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        char_embeddings = self.char_embeddings(input)   # (sentence_length * batch_size * max_word_length * embed_size)
        sentence_length, batch_size, word_length, _ = char_embeddings.shape
        char_embeddings_reshaped = char_embeddings.view((sentence_length * batch_size), word_length, self.char_embed_size).transpose(1, 2)

        x_conv = self.CNN(char_embeddings_reshaped)
        x_highway = self.Highway(x_conv)
        x_dropout = self.dropout(x_highway)
        output = x_dropout.view(sentence_length, batch_size, self.embed_size)


        return output
        ### END YOUR CODE
