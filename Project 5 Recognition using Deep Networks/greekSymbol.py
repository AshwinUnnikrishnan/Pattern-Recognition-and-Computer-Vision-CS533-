# Ashwin Unnikrishnan
# Project 5: Recognition using Deep Networks
# Date : 8 April 2022
# Greek Symbol Task 3

import lib as library
from network import greekNetwork
import torch
from torchsummary import summary

labelDict = {0 : "alpha", 1:"beta", 2:"gamma", 3:"lambda", 4:"tau"}

#loading the data : Loading greek symbol dataset
greekLoader = library.loadCustomData('greek',100)
dataN = enumerate(greekLoader)
batch_idx, (example_data, example_targets) = next(dataN)

res = []
for i in range(len(example_data)):
    res.append(labelDict[i//9])
print(res)

library.plotGraph(12,example_data, res, "GreekDataSet", True)



#loading the model : Creating a truncated model
continued_network = greekNetwork()
network_state_dict = torch.load('modelMNIST.pth')
continued_network.load_state_dict(network_state_dict)


#getting features : Projecting the greek symbols in embedding space
featuresGreek = library.test(continued_network, greekLoader, [], 'greek')

#checking things
summary(continued_network, (1, 28, 28))

res = []
for i in range(0,36,9):
    indSSD = library.sumSquareDistanceM(featuresGreek,i)
    #Getting the index of the least distance and printing the label
    res.append(labelDict[indSSD//9])
    print(labelDict[indSSD//9])


#loading the custom data : Loading greek symbol dataset
greekLoaderCustom = library.loadCustomData('customGreek',100)
dataNC = enumerate(greekLoaderCustom)
batch_idx, (eData, eTarget) = next(dataNC)

featuresGreekCustom = library.test(continued_network, greekLoaderCustom, [], 'greek')
res = []
for i in range(len(eData)):
    indSSD = library.sumSquareDistanceM(featuresGreek,i, featuresGreekCustom[i], True)
    #Getting the index of the least distance and printing the label
    res.append(labelDict[indSSD//9])
    #print(labelDict[indSSD//9])
print(res)
library.plotGraph(4,eData, res, "1_Symbol_From Each 1NN classifier", True)



#knn classifier extension

res= []
for i in range(len(eData)):
    indSSD = library.KNNclassifier(featuresGreek, featuresGreekCustom[i])
    #Getting the index of the least distance and printing the label
    #print(indSSD)
    res.append(labelDict[indSSD])
    #print(labelDict[indSSD//9])

library.plotGraph(4,eData, res, "CustomDataWritten KNN", True)
