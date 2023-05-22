## base package
import numpy as np

## torch package
import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN,self).__init__()
        self.conv1          = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
        self.BN1            = nn.BatchNorm2d(64)

        self.conv2          = nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)
        self.BN2            = nn.BatchNorm2d(128)

        self.conv3          = nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1)
        self.BN3            = nn.BatchNorm2d(256)

        self.pool           = nn.MaxPool2d(2,2)
        self.Activation     = nn.ReLU(inplace=True)
        self.dropout        = nn.Dropout(0.5,inplace=True)

        self.fc1            = nn.Linear(256*16*16,1024)
        self.fc2            = nn.Linear(1024,512)
        self.fc3            = nn.Linear(512,2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.Activation(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.BN2(x)
        x = self.Activation(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.BN3(x)
        x = self.Activation(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    

if __name__ == "__main__":
    model = CNN().to("cuda")

    x = np.arange(98304).reshape([2, 3, 128, 128])
    x_tensor = torch.Tensor(x).to("cuda")
    # model.train(x_tensor)
    pred = model.forward(x_tensor)
    cls = torch.max(pred,dim=1)
    print(pred,'\n',cls.indices)