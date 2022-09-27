from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from data_utils import map_to_tensor, prepare_data
from fsdl.lab01.text_recognizer.data.util import BaseDataset
from fsdl.lab01.text_recognizer.models.mlp import MLP
from nn_1layer_v2_pytorch import configure_opt  # , MNISTLosgistic


def fit(xtrain, ytrain, epochs, lr, bs):
    loss_history = []
    digits_to_9 = list(range(10))
    data_config = {
        "input_dims": (784,),
        "mapping": {digit: str(digit) for digit in digits_to_9},
    }
    model = MLP(data_config)
    # model = MNISTLosgitic()
    opt = configure_opt(model=model, lr=lr)
    loss_fn = cross_entropy
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
    return loss_history, pred


if __name__ == "__main__":

    xtrain, ytrain, xvalid, yvalid = prepare_data()

    xtrain, ytrain, xvalid, yvalid = map_to_tensor(
        xtrain, ytrain, xvalid, yvalid
    )
    loss_hist, _ = fit(xtrain, ytrain, 5, 0.01, 1000)
