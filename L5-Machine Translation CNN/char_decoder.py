#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        super(CharDecoder, self).__init__()  # remember to inherit the parent class
        V_vocab = len(target_vocab.char2id)
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)  # it's unidirectional
        self.char_output_projection = nn.Linear(hidden_size, V_vocab, bias=True)  # with bias
        self.decoderCharEmb = nn.Embedding(V_vocab, char_embedding_size, padding_idx=target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab

        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        ### END YOUR CODE

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        char_embedding = self.decoderCharEmb(input)  # shape of (length, batch, char_embed_size)

        output, dec_hidden = self.charDecoder(char_embedding, dec_hidden)
        # output of shape (length, batch, hidden_size), two tensors of dec_hidden have a shape of (1, batch, hidden_size)
        scores = self.char_output_projection(output)
        """ I think there is something wrong with the class notes. It's not the hidden states (h_n) that 
        the projection layer takes, but the output of the CharDecoderLSTM. """

        return scores, dec_hidden
        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch).
        Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder.
        A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        loss = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'], reduction='sum')
        """ This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class. """

        scores, dec_hidden = self.forward(char_sequence[:-1], dec_hidden)
        # scores of shape (length, batch, self.vocab_size)
        output = loss(scores.view(-1, scores.shape[-1]),
                      char_sequence[1:].contiguous().view(-1))  # Hey, python starts with zero

        """ The CrossEntropyLoss takes an input of shape (N, C) and generates an output of shape (N). 
        Therefore, we need to reshape the scores (input) and the target (for comparison).

        Besides, A contiguous array is just an array stored in an unbroken block of memory: 
        to access the next value in the array, we just move to the next memory address. """

        return output
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        batch_size = initialStates[0].shape[1]

        start_index = self.target_vocab.start_of_word
        end_index = self.target_vocab.end_of_word

        current_char = torch.tensor([start_index for _ in range(batch_size)], device=device).unsqueeze(0)  # of shape (1, batch)
        dec_hidden = initialStates
        output_word = [""] * batch_size
        output_word_flag = [True] * batch_size   # we need a flag to skip the current character in the batch

        for t in range(max_length):
            scores, dec_hidden = self.forward(current_char, dec_hidden)  # score of shape (1, batch, self.vocab_size)
            current_char = scores.argmax(dim=2)  # shape of (1, batch)

            for index, char in enumerate(current_char.squeeze(0)):  # shape of (batch)
                """ I don't know why we need detach(). """
                if output_word_flag[index]:
                    if char.item() == end_index:
                        output_word_flag[index] = False
                    else:
                        output_word[index] += self.target_vocab.id2char[char.item()]

        return output_word

        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        ### END YOUR CODE

