from pathlib import Path
import requests
import torch
import pickle
import gzip


def download_data(path):
    """

    :param path: 

    """
    url = "https://github.com/pytorch/tutorials/raw/master/static/"
    filename = "mnist.pkl.gz"

    if not (path / filename).exists():
        content = requests.get(url + filename).content
        (path / filename).open("wb").write(content)

    return path / filename

# if __name__ == "__main__":
#
#     data_path = Path("data") if Path("data").exists() else Path("../data")
#     path = data_path / "downloaded" / "vector-mnist"
#     path.mkdir(parents=True, exist_ok=True)
#
#     datafile = download_data(path)
#     print(datafile)
