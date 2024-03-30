import torch
from torch import nn, optim
import numpy as np


class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Seq2SeqLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, self.num_layers, batch_first=True
        )
        self.linear = nn.Linear(self.hidden_size, 16 * 3)

    def forward(self, input):
        output, hidden = self.lstm(input)

        # print(1, output.shape)

        output = self.linear(output)
        return output


if __name__ == "__main__":

    model = Seq2SeqLSTM(input_size=17 * 3, hidden_size=64, num_layers=2)

    pesudo_data = np.random.rand(32, 29, 17 * 3)

    input_tensor = torch.tensor(pesudo_data, dtype=torch.float)

    num_epochs = 30

    output = model(input_tensor)

    print(output.shape)
