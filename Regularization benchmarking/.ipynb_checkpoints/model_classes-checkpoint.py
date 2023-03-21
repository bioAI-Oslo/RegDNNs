import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


### MLP

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features = 32 * 32 * 3, out_features = 64)
        self.fc2 = nn.Linear(in_features = 64, out_features = 64)
        self.fc3 = nn.Linear(in_features = 64, out_features = 64)
        self.fc4 = nn.Linear(in_features = 64, out_features = 10)
    
    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim = 1)
        return x

### LENET
    
class LeNet(nn.Module):
    def __init__(self, lr, momentum):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = (5, 5))
        self.pool = nn.MaxPool2d(kernel_size = (2, 2), stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = (5, 5))
        self.fc1 = nn.Linear(in_features = 16 * 5 * 5, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 84)
        self.fc3 = nn.Linear(in_features = 84, out_features = 10)
        
        self.L = nn.CrossEntropyLoss()
        self.opt = torch.optim.SGD(self.parameters(), lr = lr, momentum = momentum)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim = 1)
        return x
    
    def loss_fn(self, x, y, l1 = False, l1_lmbd = 0.00001, l2 = False, l2_lmbd = 0.0001, soft_svb = False, soft_svb_lmbd = 0.01):
        y_pred = self(x.float())
        loss = self.L(y_pred, y)
        if l1:
            loss += l1_lmbd * self.fc1.weight.abs().sum() # L1 reg
        if l2:
            loss += (l2_lmbd * self.fc1.weight**2).sum()
        if soft_svb:
            w = self.fc1.weight
            # Main loss function term for soft SVB from Jia et al. 2019:
            w_orth = w.transpose(0,1) @ w # W^T * W
            w_orth = w_orth - torch.eye(w_orth.shape[0]) # W^T * W - I
            loss += soft_svb_lmbd * torch.linalg.norm(w_orth, ord = "fro")**2
        return loss
    
    def train_step(self, data, labels, l1 = False, l1_lmbd = 0.00001, l2 = False, l2_lmbd = 0.0001, soft_svb = False, soft_svb_lmbd = 0.01):
        self.opt.zero_grad()
        loss = self.loss_fn(data, labels.long(), l1 = l1, l1_lmbd = l1_lmbd, l2 = l2, l2_lmbd = l2_lmbd, soft_svb = soft_svb, soft_svb_lmbd = soft_svb_lmbd)
        loss.backward()
        self.opt.step()
        return loss.item()

### RESNET

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


