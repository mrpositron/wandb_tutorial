import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, layer_dims, dropout):
        super(Net, self).__init__()
        self.layer_arr = []
        for i in range(len(layer_dims) - 1):
            self.layer_arr.extend(
                [nn.Linear(layer_dims[i], layer_dims[i + 1]), 
                nn.ReLU(),
                nn.Dropout(dropout)]    
            )
        self.nn = nn.Sequential(*self.layer_arr)

    def forward(self, X):
        X = X.view(X.shape[0], -1)
        X = self.nn(X)
        return F.log_softmax(X, dim=1)


if __name__ == "__main__":
    model = Net([784, 10], 0.3)