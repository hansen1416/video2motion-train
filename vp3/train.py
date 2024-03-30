import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from lstm import Seq2SeqLSTM
from dataset import Datensatz


ds = Datensatz()


# Split the dataset into train and test sets
train_size = int(0.8 * len(ds))
test_size = len(ds) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, test_size])

batch_size = 32
seq_length = 29

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(len(train_loader), len(test_loader))


model = Seq2SeqLSTM(input_size=17 * 3, hidden_size=64, num_layers=2)

num_epochs = 300
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)


for epoch in range(num_epochs):

    for i, (input_tensor, target_tensor) in enumerate(train_loader):
        # the last batch might not be a full batch, skip it
        if input_tensor.shape[0] != batch_size:
            continue

        # print(input_tensor.shape, target_tensor.shape)

        optimizer.zero_grad()
        output = model(input_tensor.view(batch_size, seq_length, -1))
        output = output.squeeze()
        loss = criterion(output, target_tensor.view(batch_size, seq_length, -1))
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print("Epoch: {} -- Training loss (MSE) {}".format(epoch, loss.item()))
