from pathlib import Path
import requests
import torch
import pickle
import gzip
import wandb
import random


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


if __name__ == "__main__":

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
