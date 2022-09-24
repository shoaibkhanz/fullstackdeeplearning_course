import math
import random
from pathlib import Path

import torch
import wandb

from data_utils import download_data, map_to_tensor, read_mnist

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)
print(weights.shape)
print(bias.shape)

data_path = Path("data") if Path("data").exists() else Path("./data")
path = data_path / "downloaded" / "vector-mnist"
path.mkdir(parents=True, exist_ok=True)

datafile = download_data(path)
print(datafile)
xtrain, ytrain, xvalid, yvalid = read_mnist(datafile)
print(xtrain.shape)

datalist = [xtrain, ytrain, xvalid, yvalid]
xtrain, ytrain, xvalid, yvalid = map_to_tensor(xtrain, ytrain, xvalid, yvalid)
print(xtrain.shape)

idx = random.randint(0, len(xtrain))
example = xtrain[idx]

print(ytrain[idx])
wandb.Image(example.reshape(1, 28, 28)).image

# if __name__ == "__main__":
