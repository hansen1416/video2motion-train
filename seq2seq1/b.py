# Model - seq2seq model without loop over decoder
import torch
from torch import nn, optim
import numpy as np

# Generate data

sinewave = np.sin(np.arange(0, 2000, 0.1))
slices = sinewave.reshape(-1, 200)
input_tensor = torch.tensor(slices[:, :-1], dtype=torch.float).unsqueeze(2)
target_tensor = torch.tensor(slices[:, 1:], dtype=torch.float)
print(input_tensor.shape, target_tensor.shape)


class Seq2SeqB(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Seq2SeqB, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, self.num_layers, batch_first=True
        )
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, input):
        output, hidden = self.lstm(input)
        output = self.linear(output)
        return output


seq2seqB = Seq2SeqB(input_size=1, hidden_size=51, num_layers=2)


# Training- seq2seq model without loop over decoder

num_epochs = 300
criterion = nn.MSELoss()
optimizer = optim.Adam(seq2seqB.parameters(), lr=0.001)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = seq2seqB(input_tensor)
    output = output.squeeze()
    loss = criterion(output, target_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print("Epoch: {} -- Training loss (MSE) {}".format(epoch, loss.item()))
