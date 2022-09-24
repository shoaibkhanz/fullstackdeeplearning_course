import gzip
import math
import pickle
from pathlib import Path

import requests
import torch


def download_data(path):
    """

    :param path:

    """
    url = "https://github.com/pytorch/tutorials/raw/master/_static/"
    filename = "mnist.pkl.gz"

    if not (path / filename).exists():
        content = requests.get(url + filename).content
        (path / filename).open("wb").write(content)

    return path / filename


def read_mnist(path):
    with gzip.open(path, "rb") as f:
        ((xtrain, ytrain), (xvalid, yvalid), _) = pickle.load(
            f, encoding="latin-1"
        )
    return xtrain, ytrain, xvalid, yvalid


def map_to_tensor(xtrain, ytrain, xvalid, yvalid):
    xtrain, ytrain, xvalid, yvalid = map(
        torch.tensor, (xtrain, ytrain, xvalid, yvalid)
    )
    return xtrain, ytrain, xvalid, yvalid


def linear(x, weights, bias):
    return x @ weights + bias


def log_softmax(x: torch.Tensor) -> torch.Tensor:
    return x - torch.log(torch.sum(torch.exp(x), axis=1))[:, None]


def cross_entropy(pred, actual):
    return -pred[range(actual.shape[0]), actual].mean()


def mnist_wb(nclasses: int = 10) -> torch.Tensor:
    weights = torch.randn(784, nclasses) / math.sqrt(784)
    weights.requires_grad_()
    bias = torch.zeros(nclasses, requires_grad=True)
    return weights, bias


def prepare_data() -> torch.Tensor:
    data_path = (
        Path("data") if Path("data").exists() else Path("./data")
    )
    path = data_path / "downloaded" / "vector-mnist"
    path.mkdir(parents=True, exist_ok=True)

    datafile = download_data(path)
    xtrain, ytrain, xvalid, yvalid = read_mnist(datafile)
    return xtrain, ytrain, xvalid, yvalid
