# Ashwin Unnikrishnan
# Project 5: Recognition using Deep Networks
# Date : 7 April 2022
# Network Examining Task 2

from network import MyNetwork, Submodel, SubmodelTrunc
import torch
import lib as library
from torchsummary import summary

random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

batch_test = 1000
batch_train = 64

#loading the data of mnist dataset
train_loader, test_loader = library.testLoader('mnist', 1, 1)

#Loading the trained model

continued_network = MyNetwork()
network_state_dict = torch.load('modelMNIST.pth')
continued_network.load_state_dict(network_state_dict)

#printing the summary of the model
summary(continued_network, (1, 28, 28))
print('------------------Layer 1------------------')
layer1 = continued_network.conv1.weight
print("Shape of Layer 1 {0}".format(layer1.shape))
print("Weight and respective shapes of layer 1")

# visualize the first conv layer filters
library.visualizeLayer(layer1, "firstLayer")   #uncomment


examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

firstImg = example_data[0]

library.filterEffects('firstLayerFilterOutput', firstImg, layer1)  #Uncomment


#Getting the truncated 2 Layered Model
subNetwork = Submodel()
network_state_dict = torch.load('modelMNIST.pth')
subNetwork.load_state_dict(network_state_dict)
subNetwork.eval()

with torch.no_grad():
    output = subNetwork(example_data)

output = output[0]
library.drawClasOutput(output, "TruncatedOut")

#Getting the truncated 1 Layered Model

subNetworkT = SubmodelTrunc()
network_state_dictT = torch.load('modelMNIST.pth')
subNetworkT.load_state_dict(network_state_dictT)
with torch.no_grad():
    output = subNetworkT(example_data)
output = output[0]
library.drawClasOutput(output, "TruncatedOut1")
