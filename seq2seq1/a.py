import torch
from torch import nn, optim
import numpy as np

# Generate data

sinewave = np.sin(np.arange(0, 2000, 0.1))
slices = sinewave.reshape(-1, 200)
input_tensor = torch.tensor(slices[:, :-1], dtype=torch.float).unsqueeze(2)
target_tensor = torch.tensor(slices[:, 1:], dtype=torch.float)
print(input_tensor.shape, target_tensor.shape)


# Model - seq2seq model with loop over decoder


class Seq2SeqA(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, batch_size, sequence_length
    ):
        super(Seq2SeqA, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.encoder_lstm = nn.LSTM(
            self.input_size, self.hidden_size, self.num_layers, batch_first=True
        )
        self.decoder_lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, input):
        _, hidden = self.encoder_lstm(input)
        input_t = torch.zeros(self.batch_size, 1, dtype=torch.float).unsqueeze(0)
        output_tensor = torch.zeros(self.sequence_length, self.batch_size, 1)
        for t in range(self.sequence_length):
            output_t, hidden = self.decoder_lstm(input_t, hidden)
            output_t = self.linear(output_t[-1])
            input_t = output_t.unsqueeze(0)
            output_tensor[t] = output_t

        return output_tensor


seq2seqA = Seq2SeqA(
    input_size=1, hidden_size=51, num_layers=1, batch_size=100, sequence_length=199
)


# Training - seq2seq model with loop over decoder

num_epochs = 300
criterion = nn.MSELoss()
optimizer = optim.Adam(seq2seqA.parameters(), lr=0.001)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = seq2seqA(input_tensor)
    output = output.squeeze().transpose(1, 0)
    loss = criterion(output, target_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print("Epoch: {} -- Training loss (MSE) {}".format(epoch, loss.item()))
