# Ashwin Unnikrishnan
# Project 5: Recognition using Deep Networks
# Date : 8 April 2022
# Stores Different types of models/networks

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MyNetwork(nn.Module):
    def __init__(self, prate=0.5):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d(p=prate)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.max_pool(self.conv1(x)))
        x = F.relu(self.max_pool(self.dropout(self.conv2(x))))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, -1)

class Submodel(MyNetwork):
    '''
    Truncated ot have just conv1 and conv2 layer
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the forward method
    def forward( self, x ):
        x = F.relu(self.max_pool(self.conv1(x)))  # relu on max pooled results of conv1
        x = F.relu(self.max_pool(self.dropout(self.conv2(x))))  # relu on max pooled results of dropout of conv2
        return x

class SubmodelTrunc(MyNetwork):
    '''
    Truncated the conv2 layer
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the forward method
    def forward( self, x ):
        x = F.relu(self.max_pool(self.conv1(x)))
        return x

class greekNetwork(MyNetwork):
    '''
    Removed the final layer gets 50 features

    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = F.relu(self.max_pool(self.conv1(x)))
        x = F.relu(self.max_pool(self.dropout(self.conv2(x))))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return F.log_softmax(x, -1)

'''class gaborNetwork(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(gabor(x),2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)),2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, -1)
    '''
