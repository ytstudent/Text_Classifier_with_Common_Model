import torch
from torch import nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocabulary_size, embed_dim, num_filters, filter_sizes, dropout, num_classes):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocabulary_size, embed_dim)
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.convs = nn.ModuleList([nn.Conv2d(1, self.num_filters, (k, embed_dim)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.num_filters * len(filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embedding(x.long())  # (N,V,E)   eg: [65,50,300]
        x = x.unsqueeze(1)  # (N,Ci,V,E)    eg: [54,1,50,300]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # len(Ks)*(N, filter_sizes, V)
        # eg :F.relu(conv(x)).squeeze(3).size() = ([65, 16, 49],[65, 16, 48],[65, 16, 47])

        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N, filter_sizes)   eg: [65,16]
        x = torch.cat(x, 1)  # (N, filter_sizes*len(Ks))  eg:[65,48]
        x = self.dropout(x)
        logit = self.fc(x)
        return logit
