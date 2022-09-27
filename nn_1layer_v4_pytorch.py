from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from data_utils import map_to_tensor, prepare_data
from fsdl.lab01.text_recognizer.data.util import BaseDataset
from nn_1layer_v2_pytorch import MNISTLosgitic, configure_opt

if __name__ == "__main__":

    bs = 1000
    epochs = 9
    lr = 0.01
    loss_fn = cross_entropy
    loss_history = []
    xtrain, ytrain, xvalid, yvalid = prepare_data()

    xtrain, ytrain, xvalid, yvalid = map_to_tensor(
        xtrain, ytrain, xvalid, yvalid
    )
    model = MNISTLosgitic()
    opt = configure_opt(model=model, lr=lr)
    train_ds = BaseDataset(xtrain, ytrain)
    train_dataloader = DataLoader(train_ds, batch_size=bs)
    for epoch in range(epochs):
        for xb, yb in train_dataloader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            loss_history.append(loss)
            print(f"epoch#: {epoch}, loss: {loss}")
            opt.step()
            opt.zero_grad()
