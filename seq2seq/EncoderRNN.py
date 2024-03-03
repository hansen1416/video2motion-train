import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    """
    We initialize this module to be bidirectional,
    meaning that we have two independent GRUs:
    one that iterates through the sequences in chronological order, and another that iterates in reverse order.
    We ultimately return the sum of these two GRUs’ outputs.

    To batch variable-length sentences,
    we allow a maximum of MAX_LENGTH tokens in a sentence,
    and all sentences in the batch that have less than MAX_LENGTH tokens are padded at the end
    with our dedicated PAD_token tokens.

    Encoding the input sequence is straightforward:
    simply forward the entire sequence tensor and its corresponding lengths vector to the encoder.
    It is important to note that this module only deals with one input sequence at a time, NOT batches of sequences.
    Therefore, when the constant 1 is used for declaring tensor sizes, this corresponds to a batch size of 1.
    """

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the ``input_size`` and ``hidden_size`` parameters are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            dropout=(0 if n_layers == 1 else dropout),
            bidirectional=True,
        )

    def forward(self, input_seq, input_lengths, hidden=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]
        # Return output and final hidden state
        return outputs, hidden
