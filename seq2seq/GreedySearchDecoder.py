import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import unicodedata
import numpy as np

device = torch.device("cpu")

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class GreedySearchDecoder(nn.Module):
    """
    To decode a given decoder output, we must iteratively run forward passes through our decoder model,
    which outputs softmax scores corresponding to the probability of each word being the correct next word in the decoded sequence.
    We initialize the decoder_input to a tensor containing an SOS_token.

    After each pass through the decoder,
    we greedily append the word with the highest softmax probability to the decoded_words list.
    We also use this word as the decoder_input for the next iteration.

    The decoding process terminates either if the decoded_words list has reached a length of MAX_LENGTH
    or if the predicted word is the EOS_token.
    """

    def __init__(self, encoder, decoder, decoder_n_layers):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._device = device
        self._SOS_token = SOS_token
        self._decoder_n_layers = decoder_n_layers

    __constants__ = ["_device", "_SOS_token", "_decoder_n_layers"]

    def forward(
        self, input_seq: torch.Tensor, input_length: torch.Tensor, max_length: int
    ):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[: self._decoder_n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = (
            torch.ones(1, 1, device=self._device, dtype=torch.long) * self._SOS_token
        )
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self._device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self._device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores
