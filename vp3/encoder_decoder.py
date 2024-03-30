from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 29


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.GRU(input_dim, hidden_dim, n_layers, dropout=0, batch_first=True)

    def forward(self, data):
        # src = [src length, batch size]

        outputs, hidden = self.rnn(data)

        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.GRU(input_dim, hidden_dim, n_layers, dropout=0, batch_first=True)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):

        batch_size = encoder_outputs.size(0)

        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=device
        ).fill_(0)

        decoder_hidden = encoder_hidden

        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden
            )
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                # detach from history as input
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        # We return `None` for consistency in the training loop
        return (
            decoder_outputs,
            decoder_hidden,
            None,
        )

    def forward_step(self, input, hidden):

        output, hidden = self.gru(input, hidden)

        return output, hidden


if __name__ == "__main__":

    from dataset import Datensatz

    ds = Datensatz()

    batch_size = 32

    features, targets = ds[100 : 100 + batch_size]

    print(features.shape, targets.shape)

    encoder = Encoder(input_dim=29, hidden_dim=128, n_layers=4)

    # hidden.shape = [4, batch_size, 128]
    # cell.shape = [4, batch_size, 128]
    hidden, cell = encoder(features.view(batch_size, 29, -1))

    print(hidden.shape, cell.shape)
