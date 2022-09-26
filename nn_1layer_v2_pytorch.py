import math

import torch
import torch.nn.functional as f
from torch import nn, optim

from data_utils import accuracy, map_to_tensor, prepare_data


class MNISTLosgitic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)


def configure_opt(model: nn.Module, lr: float) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=lr)


def fit(xtrain, ytrain):

    loss_history = []
    bs = 1000
    lr = 0.01
    n = xtrain.shape[0]
    epochs = 5

    model = MNISTLosgitic()
    opt = configure_opt(model=model, lr=lr)
    for epoch in range(epochs):
        for ii in range((n - 1) // bs + 1):
            start_index = ii * bs
            end_index = start_index + bs
            xb = xtrain[start_index:end_index]
            yb = ytrain[start_index:end_index]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            loss_history.append(loss)

            print(f"epoch#: {epoch}, batch#: {ii}, loss: {loss}")

            opt.step()
            opt.zero_grad()
    return loss_history, pred


if __name__ == "__main__":

    xtrain, ytrain, xvalid, yvalid = prepare_data()

    xtrain, ytrain, xvalid, yvalid = map_to_tensor(
        xtrain, ytrain, xvalid, yvalid
    )
    loss_fn = f.cross_entropy

    loss_hist, _ = fit(xtrain, ytrain)
