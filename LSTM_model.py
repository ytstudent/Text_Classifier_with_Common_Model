import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, vocabulary_size, embed_dim, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocabulary_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, 1)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, real_len):  # x.size is [15, 60]
        x = self.embedding(x.long())  # after the embedding function ,x.size is [15, 60 ,300]
        batch = x.size()[0]
        real_len = torch.tensor(real_len)  # real_len.size is [15]

        # Forward
        out, _ = self.lstm(x)  # out: out.size is [15, 60, 64]

        index = torch.arange(batch).long()
        state = out[index, real_len - 1, :]  # it's hard to understand!

        out = self.fc(state)
        return out
