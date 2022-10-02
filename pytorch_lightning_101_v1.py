import logging
import textwrap
import traceback
from typing import Tuple

import pytorch_lightning as pl
import torch

from fsdl.lab02.text_recognizer.lit_models import BaseLitModel


# class LinearRegression(pl.LightningModule):
#
#     def __init__(self):
#         super().__init__()
#
#         self.model = torch.nn.Linear(in_features=1,out_features=1)
#
#     def forward(self,xs):
#         return self.model(xs)
class LinearRegression(pl.LightningModule):
    def __init__(self):
        super().__init__()  # just like in torch.nn.Module, we need to call the parent class __init__

        # attach torch.nn.Modules as top level attributes during init, just like in a torch.nn.Module
        self.model = torch.nn.Linear(in_features=1, out_features=1)
        # we like to define the entire model as one torch.nn.Module -- typically in a separate class

    # optionally, define a forward method
    def forward(self, xs):
        return self.model(
            xs
        )  # we like to just call the model's forward method

    def training_step(
        self: pl.LightningModule,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:

        xs, ys = batch
        preds = self(xs)
        loss = torch.nn.functional.mse_loss(preds, ys)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=3e-4)


class CorrelatedDataset(torch.utils.data.Dataset):
    def __init__(self, N=10_000):
        self.N = N
        self.xs = torch.randn(size=(N, 1))
        self.ys = torch.randn_like(self.xs) + self.xs

    def __getitem__(self, idx):
        return (self.xs[idx], self.ys[idx])

    def __len__(self):
        return self.N


if __name__ == "__main__":

    data = CorrelatedDataset()
    train_dataloader = torch.utils.data.DataLoader(
        data, batch_size=32, num_workers=1
    )
    trainer = pl.Trainer(
        max_epochs=20, gpus=int(torch.cuda.is_available())
    )
    model = LinearRegression()
    print(
        f"Before training: Loss: {torch.mean(torch.square(model(data.xs)- data.ys)).item()}"
    )
    trainer.fit(model=model, train_dataloaders=train_dataloader)
    print(
        f"After training: Loss: {torch.mean(torch.square(model(data.xs)- data.ys)).item()}"
    )


# try:
#     logging.getLogger("Pytorch Lightning").setLevel(logging.ERROR)
#     model = LinearRegression()
#
#     trainer = pl.Trainer(gpus=int(torch.cuda.is_available()),max_epochs=1)
#     trainer.fit(model=model)
#
# # except pl.utilities.exceptions.MisconfigurationException as error:
# except ValueError as error:
#     print("Error:", *textwrap.wrap(str(error), 80), sep="\n\t")  # show the error without raising it
#
# finally:
#     logging.getLogger("Pytorch Lightning").setLevel(logging.INFO)
