import math
from typing import Tuple

import pytorch_lightning as pl
import torch


class CorrelatedDataset(torch.utils.data.Dataset):
    def __init__(self, N):
        self.N = N
        self.xs = torch.randn(N, 1)
        self.ys = torch.randn_like(self.xs) + self.xs

    def __getitem__(self, idx):
        return (self.xs[idx], self.ys[idx])

    def __len__(self):
        return self.N


class CorrelatedDataModule(pl.LightningModule):
    def __init__(self, size=10_000, train_frac=0.8, batch_size=34):
        super().__init__()

        self.size = size
        self.train_frac, self.val_frac = train_frac, 1 - train_frac
        self.train_indices = list(
            range(math.floor(self.size * train_frac))
        )
        self.valid_indices = list(
            range(self.train_indices[-1], self.size)
        )
        self.batch_size = batch_size
        self.dataset = CorrelatedDataset(N=self.size)

    def set_up(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = torch.utils.data.Subset(
                self.dataset, self.train_indices
            )
            self.valid_dataset = torch.utils.data.Subset(
                self.dataset, self.valid_dataset
            )

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset, self.batch_size
        )

    def valid_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset, self.batch_size
        )


class LinearRegression(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = torch.nn.Linear(in_features=1, out_features=1)

    def forward(self, xs):
        return self.model(xs)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        xs, ys = batch
        pred = self.model(xs)
        loss = torch.nn.functional.mse_loss(pred, ys)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)


if __name__ == "__main__":

    data_module = CorrelatedDataModule()
    model = LinearRegression()
    dataset = data_module.dataset
    print(
        f"loss before training: {torch.mean(model(dataset.xs) - dataset.ys)}"
    )

    trainer = pl.Trainer(
        max_epochs=10, gpus=int(torch.cuda.is_available())
    )
    trainer.fit(model=model, datamodule=data_module)

    print(
        f"loss after training: {torch.mean(model(dataset.xs) - dataset.ys)}"
    )
