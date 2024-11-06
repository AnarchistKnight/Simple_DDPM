from torch import nn


class MLP(nn.Module):
    def __init__(self, dim_model, dim_ffd_hidden, activation, dropout):
        super().__init__()
        self.linear1 = nn.Linear(dim_model, dim_ffd_hidden)
        self.activation = activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ffd_hidden, dim_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x