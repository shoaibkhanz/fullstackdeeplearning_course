import random
from pathlib import Path

import torch
import wandb

from data_utils import (
    cross_entropy,
    linear,
    log_softmax,
    map_to_tensor,
    mnist_wb,
    prepare_data,
)

if __name__ == "__main__":

    xtrain, ytrain, xvalid, yvalid = prepare_data()

    xtrain, ytrain, xvalid, yvalid = map_to_tensor(
        xtrain, ytrain, xvalid, yvalid
    )
    print(xtrain.shape)

    idx = random.randint(0, len(xtrain))
    example = xtrain[idx]

    print(ytrain[idx])
    wandb.Image(example.reshape(1, 28, 28)).image

    bs = 64
    xb = xtrain[:bs]
    yb = ytrain[:bs]
    weights, bias = mnist_wb()
    preds = log_softmax(linear(xb, weights=weights, bias=bias))

    loss_func = cross_entropy
    loss = loss_func(preds, yb)
    print(f"loss: {loss}")
    print(bias.grad)
