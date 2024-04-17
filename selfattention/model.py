# https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, d_in, d_out_kq, d_out_v):
        """
        Args:
            d_in: int, embedding_size, we use 3 as the embedding size, only for demonstration,
                in practice, use a larger embedding size. eg. Llama 2 utilizes embedding sizes of 4,096

            d_out_kq: int, the number of elements in the query and key vectors, d_q = d_k
                Since we are computing the dot-product between the query and key vectors,
                these two vectors have to contain the same number of elements (d_q = d_k) `d_out_kq`

            d_out_v: int, the number of elements in the value vector v(i),
                In many LLMs, we use the same size for the value vectors such that d_q = d_k = d_v.
                However, the number of elements in the value vector v(i),
                which determines the size of the resulting context vector, can be arbitrary.
        """
        super().__init__()
        self.d_out_kq = d_out_kq

        # (embedding_size, d_out_kq)
        # self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_query = nn.Linear(d_in, d_out_kq)
        # (embedding_size, d_out_kq)
        # self.W_key = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key = nn.Linear(d_in, d_out_kq)
        # (embedding_size, d_out_v)
        # self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))
        self.W_value = nn.Linear(d_in, d_out_v)

    def forward(self, x):

        # each item in `queries` is the queries weights for each word in the sentence
        # represents what information a specific element in the sequence needs from others. therefore keys.T
        queries = self.W_query(x)

        # each item in `keys` is the keys weights for each word in the sentence
        # represents what information each element in the sequence can provide.
        keys = self.W_key(x)

        # each item in `values` is the values weights for each word in the sentence
        # holds the actual information of each element.
        values = self.W_value(x)

        # (batch_size, sentence_length, embedding_size) @ (embedding_size, d_out_kq) = (batch_size, sentence_length, d_out_kq)
        # (batch_size, sentence_length, d_out_kq)
        # print(queries.shape, keys.shape, values.shape)

        # attention score $\omega_{i,j} = q^{(i)} k^{(j)}$
        # (batch_size, sentence_length, d_out_kq) @ (batch_size, d_out_kq, sentence_length) = (batch_size, sentence_length, sentence_length)
        attn_scores = queries @ keys.transpose(-2, -1)

        # to obtain the normalized attention weights, α (alpha),
        # by applying the softmax function. Additionally, 1/√{d_k} is used to scale $\omega$
        # before normalizing it through the softmax function
        # The scaling by d_k ensures that the Euclidean length of the weight vectors will be approximately in the same magnitude.
        # dim=-1. This ensures that the attention weights for each element (represented by rows in the tensor) sum up to 1.
        # (batch_size, sentence_length, sentence_length)
        attn_weights = torch.softmax(attn_scores / self.d_out_kq**0.5, dim=-1)

        # the context vector z^(i), which is an attention-weighted version of our original query input x^(i),
        # including all the other input elements as its context via the attention weights:
        # (batch_size, sentence_length, sentence_length) @ (batch_size, sentence_length, d_out_v) = (batch_size, sentence_length, d_out_v)
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    """
    each attention head in multi-head attention can potentially learn to focus on different parts of the input sequence,
    capturing various aspects or relationships within the data.
    This diversity in representation is key to the success of multi-head attention.

    Multi-head attention can also be more efficient, especially in terms of parallel computation.
    Each head can be processed independently,
    making it well-suited for modern hardware accelerators like GPUs or TPUs that excel at parallel processing.

    eg. the 7B Llama 2 model uses 32 attention heads.
    """

    def __init__(
        self, input_seq_len, output_seq_len, d_in, d_out_kq, d_out_v, num_heads
    ):
        super().__init__()
        # each self-attention head will have its own set of weight matrices, they work in parallel
        self.heads = nn.ModuleList(
            [SelfAttention(d_in, d_out_kq, d_out_v) for _ in range(num_heads)]
        )

        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.d_in = d_in
        self.d_out_kq = d_out_kq
        self.d_out_v = d_out_v

        fc1_in = int(input_seq_len * d_out_v)
        fc1_out = int(input_seq_len * d_out_v // 2)
        fc2_out = input_seq_len

        # print(fc1_in, fc1_out, fc2_in, fc2_out)

        self.fc1 = nn.ModuleList([nn.Linear(fc1_in, fc1_out) for _ in range(num_heads)])
        self.activ = nn.Tanh()
        self.fc2 = nn.ModuleList(
            [nn.Linear(fc1_out, fc2_out) for _ in range(num_heads)]
        )

        # target is of shape (12, 3)
        self.fc3 = nn.Linear(fc2_out * num_heads, output_seq_len * d_in)

    def forward(self, x):

        x_linear_heads = []

        for i, head in enumerate(self.heads):
            xi = head(x)
            # flatten for each batch
            xi = xi.view(-1, self.input_seq_len * self.d_out_v)

            xi = self.fc1[i](xi)
            xi = self.activ(xi)
            xi = self.fc2[i](xi)

            x_linear_heads.append(xi)

        # (input_seq_len * num_heads)
        x = torch.cat(x_linear_heads, dim=-1)
        x = self.activ(x)
        x = self.fc3(x)

        return x.reshape(-1, self.output_seq_len, self.d_in)


if __name__ == "__main__":
    torch.manual_seed(46)

    sentence = "Life is short, eat dessert first"

    dc = {s: i for i, s in enumerate(sorted(sentence.replace(",", "").split()))}

    print(dc)

    sentence_int = torch.tensor([dc[s] for s in sentence.replace(",", "").split()])
    print(sentence_int)

    vocab_size = 50_000

    # here use 3 as the embedding size, only for demonstration, in practice, use a larger embedding size
    # eg. Llama 2 utilizes embedding sizes of 4,096
    embed = torch.nn.Embedding(vocab_size, 3)
    # torch.Size([6, 3])
    embedded_sentence = embed(sentence_int).detach()

    # (8, 6, 3)
    embedded_sentence_batch = torch.tensor(
        np.array([embedded_sentence for _ in range(8)])
    )

    # d_in, d_out_kq, d_out_v = 3, 2, 4

    # sa = SelfAttention(d_in, d_out_kq, d_out_v)

    # res = sa(embedded_sentence_batch)

    # print(res, res.shape)

    input_seq_len, output_seq_len = 6, 12

    d_in, d_out_kq, d_out_v, num_heads = 3, 2, 4, 5

    mha = MultiHeadAttentionWrapper(
        input_seq_len, output_seq_len, d_in, d_out_kq, d_out_v, num_heads
    )

    res = mha(embedded_sentence)

    print("MultiHeadAttentionWrapper:")
    print(res, res.shape)
