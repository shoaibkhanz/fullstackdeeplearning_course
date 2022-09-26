from torch.nn.functional import cross_entropy

from data_utils import map_to_tensor, prepare_data
from fsdl.lab01.text_recognizer.data.util import BaseDataset
from nn_1layer_v2_pytorch import MNISTLosgitic, configure_opt

if __name__ == "__main__":

    bs = 1000
    epochs = 4
    lr = 0.01
    loss_fn = cross_entropy
    xtrain, ytrain, xvalid, yvalid = prepare_data()

    xtrain, ytrain, xvalid, yvalid = map_to_tensor(
        xtrain, ytrain, xvalid, yvalid
    )
    train_ds = BaseDataset(xtrain, ytrain)
    n = len(train_ds.data)
    model = MNISTLosgitic()
    opt = configure_opt(model, lr=lr)

    for epoch in range(epochs):
        for ii in range((n - 1) // bs + 1):
            xb, yb = train_ds[ii * bs : ii * bs + bs]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()

            print(f"epoch#: {epoch}, batch#: {ii}, loss: {loss}")
            opt.step()
            opt.zero_grad()
