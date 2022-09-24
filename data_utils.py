from pathlib import Path
import requests
import torch
import pickle
import gzip


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
        ((xtrain, ytrain), (xvalid, yvalid), _) = pickle.load(f, encoding="latin-1")
    return xtrain, ytrain, xvalid, yvalid

def map_to_tensor(xtrain, ytrain, xvalid, yvalid):
    xtrain, ytrain, xvalid, yvalid = map(torch.tensor, (xtrain, ytrain, xvalid, yvalid))
    return xtrain, ytrain, xvalid, yvalid

