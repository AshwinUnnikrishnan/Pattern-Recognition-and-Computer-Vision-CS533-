# Ashwin Unnikrishnan
# Project 5: Recognition using Deep Networks
# Date : 7 April 2022
# loading model and checking

from network import gaborNetwork
import torch
import torch.optim as optim
import lib as library
import torch.utils.data as data_utils
import torch.nn.functional as F

random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


batch_test = 1000
batch_train = 64

#loading the data of mnist dataset
train_loader, test_loader = library.testLoader('mnist', batch_test, batch_train)

#Loading the trained model

continued_network = gaborNetwork()
network_state_dict = torch.load('modelMNIST.pth')
continued_network.load_state_dict(network_state_dict)


print("Taking the first batch of Test Data")
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)


#using the model to predict the classification
output = continued_network(example_data)
output = [ [float('%.2f' % elem) for elem in o] for o in output ]      #Converting the output to 2 decimal points

res = []
for i in range(len(output)):
    print("Image {0} and values are {1}".format(i+1, output[i]))
    res.append(library.retMaxIndex(output[i])
)
res = torch.tensor(res)
print(res)
print(example_targets)
library.plotGraph(9, example_data, res, "Predicted 10 Data") #Need to uncomment




dataLoader = library.loadCustomData('DigitRecognition')

prediction = library.test(continued_network, dataLoader, [], True)
dataN = enumerate(dataLoader)
batch_idx, (example_data, example_targets) = next(dataN)

library.plotGraph(10,example_data, prediction, "CustomDatasetDigit")
