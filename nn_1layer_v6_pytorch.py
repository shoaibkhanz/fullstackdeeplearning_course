import gzip
import pickle
from pathlib import Path

import requests
import torch
from torch import nn
from torch.nn.functional import cross_entropy, dropout, relu
from torch.utils.data import DataLoader

# from data_utils import map_to_tensor, prepare_data
from fsdl.lab01.text_recognizer.data.util import BaseDataset
from nn_1layer_v2_pytorch import configure_opt  # , MNISTLosgistic


class MNISTDataModule:

    url = "https://github.com/pytorch/tutorials/raw/master/_static/"
    filename = "mnist.pkl.gz"

    def __init__(self, dir, bs=128):

        self.dir = dir
        self.bs = bs
        self.path = self.dir / self.filename

    def prepare_data(self):
        if not (self.path).exists():
            content = requests.get(self.url + self.filename).content
            self.path.open("wb").write(content)

    def set_up(self):
        with gzip.open(self.path, "rb") as f:
            ((xtrain, ytrain), (xvalid, yvalid), _) = pickle.load(
                f, encoding="latin-1"
            )

        self.xtrain, self.ytrain, self.xvalid, self.yvalid = map(
            torch.tensor, (xtrain, ytrain, xvalid, yvalid)
        )

    def train_dataloader(self):
        train_ds = BaseDataset(self.xtrain, self.ytrain)
        return DataLoader(train_ds, batch_size=self.bs, shuffle=True)

    def val_dataloader(self):
        val_ds = BaseDataset(self.xtrain, self.ytrain)
        return DataLoader(val_ds, batch_size=self.bs, shuffle=False)


class MNISTsimplemodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = relu(x)
        x = dropout(x)
        x = self.fc2(x)
        x = relu(x)
        x = self.fc3(x)
        return x

    def fit(self: nn.Module, dataloader, epochs: int):

        dataloader.prepare_data()
        dataloader.set_up()

        train_dl = dataloader.train_dataloader()
        valid_dl = dataloader.val_dataloader()

        loss_fn = cross_entropy
        opt = configure_opt(self, lr=0.01)
        train_loss_hist = []
        valid_loss_hist = []

        self.eval()
        with torch.no_grad():
            val_loss = sum(
                [loss_fn(self(xb), yb) for xb, yb in valid_dl]
            )
            val_loss = val_loss / len(valid_dl)
            print(f"validation loss before training: {val_loss}")

        for epoch in range(epochs):
            self.train()
            for xb, yb in train_dl:
                pred = self(xb)
                loss = loss_fn(pred, yb)
                train_loss_hist.append(loss)
                loss.backward()
                print(f"epoch#: {epoch}, train_loss: {loss}")

                opt.step()
                opt.zero_grad()

            self.eval()
            with torch.no_grad():
                val_loss = sum(
                    loss_fn(self(xb), yb) for xb, yb in valid_dl
                )
                val_loss = val_loss / len(valid_dl)
                valid_loss_hist.append(val_loss)
            print(f"epoch#: {epoch}, valid_loss: {val_loss}")
        return train_loss_hist, valid_loss_hist


if __name__ == "__main__":

    data = MNISTDataModule(dir=Path("data"))
    model = MNISTsimplemodel()
    tloss, vloss = model.fit(data, epochs=5)
