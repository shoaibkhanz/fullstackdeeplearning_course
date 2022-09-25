from data_utils import prepare_data,accuracy,map_to_tensor
import math
import torch
import torch.nn.functional as f
from torch import nn



class MNISTLogistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights =nn.Parameter(torch.randn(784,10)/math.sqrt(784))
        self.bias =nn.Parameter(torch.zeros(10))

    def forward(self, xb:torch.Tensor):
        return xb@self.weights + self.bias



def fit(xtrain, ytrain):
    
    loss_history=[] 
    n=xtrain.shape[0]
    lr=0.01
    epochs=5
    bs=1000
    for epoch in range(epochs):
        for ii in range((n-1)//bs+1):
            start_index = ii*bs
            end_index = start_index+bs
            xb = xtrain[start_index:end_index]
            yb = ytrain[start_index:end_index]
            pred = model(xb)
            loss= loss_func(pred,yb)
            loss_history.append(loss)
            
            print(f"epoch#: {epoch}, batch#: {ii}, loss: {loss}")

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -=p.grad*lr
                model.zero_grad()

    return loss_history,pred


if __name__=="__main__":

    xtrain, ytrain, xvalid, yvalid = prepare_data()

    xtrain, ytrain, xvalid, yvalid = map_to_tensor(
    xtrain, ytrain, xvalid, yvalid
    )
    loss_func = f.cross_entropy

    model = MNISTLogistic()
    print(model(xtrain))

    # loss = loss_func(model(xtrain),ytrain)
    # loss.backward()
    #
    # print("#"*20,"weights","#"*20,"\n",model.weights.grad[::17,::2])
    # print(*list(model.parameters()),sep="\n")

    print("calling fit function")
    loss_hist, _ =fit(xtrain,ytrain)
    print("model training finished")
    
    print("computing accuracy")
    acc = accuracy(model(xtrain),ytrain)
    print(f"accuracy: {acc}")

