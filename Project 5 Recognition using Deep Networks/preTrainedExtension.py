# Ashwin Unnikrishnan
# Project 5: Recognition using Deep Networks
# Date : 8 April 2022
# AlexNet example


import torch
import torch.optim as optim
import lib as library
import torch.utils.data as data_utils
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
from matplotlib import pyplot as plt
import cv2
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from torchvision.models import alexnet

random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

batch_test = 1000
batch_train = 64

#loading the data of mnist dataset

#Loading the trained model

continued_network = alexnet(pretrained=True)
#network_state_dict = torch.load('modelMNIST.pth')
#continued_network.load_state_dict(network_state_dict)
print(continued_network)

greekLoader = library.loadCustomDataModelCheck('greek',100)

examples = enumerate(greekLoader)
batch_idx, (example_data, example_targets) = next(examples)

#print(example_data.shape)
summary(continued_network, (3, 224, 224))

newLayer1 = continued_network.features[0].weight
firstImg = example_data[0]
#library.visualizeLayer(newLayer1, "firstLayer")   #uncomment




examples = enumerate(greekLoader)
batch_idx, (example_data, example_targets) = next(examples)

firstImg = example_data[0]

library.filterEffects('firstLayerFilterOutput', firstImg, newLayer1[0:10,:,:,:])  #Uncomment








newLayer1 = continued_network.features[3].weight
firstImg = example_data[0]
#library.visualizeLayer(newLayer1, "firstLayer")   #uncomment




examples = enumerate(greekLoader)
batch_idx, (example_data, example_targets) = next(examples)

firstImg = example_data[0]

library.filterEffects('firstLayerFilterOutput', firstImg, newLayer1[0:10,:,:,:])  #Uncomment
