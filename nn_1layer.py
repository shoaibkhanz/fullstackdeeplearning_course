import torch

from data_utils import (
    accuracy,
    cross_entropy,
    linear,
    log_softmax,
    map_to_tensor,
    mnist_wb,
    prepare_data,
)

if __name__ == "__main__":

    xtrain, ytrain, xvalid, yvalid = prepare_data()

    xtrain, ytrain, xvalid, yvalid = map_to_tensor(
        xtrain, ytrain, xvalid, yvalid
    )
    epochs = 10
    lr = 0.05
    bs = 5000
    n = xtrain.shape[0]
    loss_hist = []
    weights, bias = mnist_wb()
    for epoch in range(epochs):
        for ii in range((n - 1) // bs + 1):
            start_idx = ii * bs
            end_idx = start_idx + bs
            xb = xtrain[start_idx:end_idx]
            yb = ytrain[start_idx:end_idx]
            print(torch.bincount(yb))
            pred = log_softmax(linear(xb, weights, bias))
            loss = cross_entropy(pred, yb)
            loss.backward()
            loss_hist.append(loss)
            print(f"loss: {loss}, epoch: {epoch}, batch: {ii}")
            with torch.no_grad():
                weights -= weights.grad * lr
                bias -= bias.grad * lr

                weights.grad.zero_()
                bias.grad.zero_()
    print(
        f"final loss: {cross_entropy(pred,yb)}, final accuracy: {accuracy(pred,yb)}"
    )
