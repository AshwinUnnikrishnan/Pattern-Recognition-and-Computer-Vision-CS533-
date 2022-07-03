# Ashwin Unnikrishnan
# Project 5: Recognition using Deep Networks
# Date : 7 April 2022
# contains all library functions

import torch
import torchvision
import sys
from matplotlib import pyplot as plt
from torchviz import make_dot
import torch.nn.functional as F
import cv2
import numpy as np

random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


def testLoader(dataset, batch_test, batch_train):
    '''
    Based on the user given dataset string returns the test and train loader objects
    :param dataset: string data telling which dataset to load
    :param batch_test: size of the test data
    :return: loader object of train and test
    '''
    if dataset == 'mnist':          # Can add other mnist dataset too
        train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(dataset, train=True, download=True,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])),batch_size=batch_train, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(dataset, train=False, download=True,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])),batch_size=batch_test, shuffle=False)

    return train_loader, test_loader

def plotGraph(ran, data, target, title, targetString=False):
    '''
    Plots the graph for the given data, target and the range of values
    :param ran: range of data to plot
    :param data: the example data image set
    :param target: the corresponding ground truth set
    :param title: title of the graph
    :return: None
    '''
    heightG = (ran+2)//3
    plt.figure(title)

    for i in range(ran):
        plt.subplot(heightG,3, i+1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        if targetString == True:
            plt.title("Ground Truth %s" % (target[i]))
        else:
            plt.title("Ground Truth %d" % (target[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def visualizeNetwork(netw, x):
    '''
    To visualize the neural network
    :param netw: the neural network to visualize
    :param x: the data input of the network
    :return: None
    '''
    y = netw(x)
    MyConvNetVis = make_dot(y, params=dict(list(netw.named_parameters()) + [('x', x)]))
    MyConvNetVis.format = "png"
    MyConvNetVis.directory = "data"
    MyConvNetVis.view()

def train(epoch, network, optimizer, log_interval, train_loader, modelName, train_losses, train_counter):
    '''
    Training and storing the model
    :param epoch: number of times we need to run
    :param network: object of the neural network we want to train on
    :param optimizer: optimizer function gradient
    :param log_interval: interval
    :param train_loader: loader for the training data
    :param modelName: name of the dataset being trained for
    :return: the trainlosses and traincounter
    '''
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx % log_interval) == 0:
          #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))       #commenting for task 4
          train_losses.append(loss.item())
          train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
          torch.save(network.state_dict(), 'model' + modelName +'.pth')
          torch.save(optimizer.state_dict(), 'optimizer'+ modelName +'.pth')
    return train_losses, train_counter

def test(network, test_loader, test_losses, test=False):
    '''
    Testing the model against the test data
    :param network: The neural network trained
    :param test_loader: loader for loading the test data
    :param test_losses: to store the loss
    :param test: if just to get result of data run then set it to True
    :return: test_losses and prediction results
    '''
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    if test==False:
        return test_losses
    if test == 'greek':
        return output
    if test == 'accuracy':
        return test_losses,(100. * correct / len(test_loader.dataset))
    return pred


def evalPerformance(train_counter, train_losses, test_counter, test_losses, save=False):
    '''
    Plot the graph of the losses of 5 epochs
    :param train_counter: training counter
    :param train_losses: stored training losses
    :param test_counter: testing counter
    :param test_losses: stored testing losses
    :return: None
    '''
    plt.figure('Performance')
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    if save != False:
        plt.savefig('ownExperiment/Performance'+ save +'.png')
    #plt.show()

def retMaxIndex(inpList):
    '''
    Given a list return the index of the maximum element
    :param inpList: input list
    :return:
    '''
    maxValI = 0
    maxVal = inpList[0]
    for i in range(1,len(inpList)):
        if inpList[i] > maxVal:
            maxValI = i
            maxVal = inpList[i]
    return maxValI

def visualizeLayer(layer, layerName):
    '''
    function to visualize a particular layer of the neural network, given the layer it plots all the filters
    :param layer: the hidden layer to be plotted
    :param layerName: name of the layer to print on the plt
    :return: None
    '''
    plt.figure(layerName)
    t = len(layer)
    for i, filter in enumerate(layer):
        plt.subplot(t//3 +1, 4, i + 1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.tight_layout()
        plt.imshow(filter[0, :, :].detach())
        plt.axis('off')
        plt.savefig(layerName+'filter.png')
        plt.xticks([])
        plt.yticks([])
    plt.show()

def filterEffects(filterName, img, filters):
    '''
    To visualize the filters and the effect of the filter on the incoming data
    :param filterName: name of the filter to put in the plot
    :param img: image to check the filtres against
    :param filters: all the filters to check image against
    :return: None
    '''
    values = filters.shape[0]
    with torch.no_grad():
        plt.figure(filterName)
        height = 5
        width = ((values*2)+height)//height
        if width%2 == 1:
            width += 1
        for i in range(values):
            resulting_image = cv2.filter2D(img.permute(1, 2, 0).numpy(), -1, filters[i, 0].numpy())
            plt.tight_layout()
            for j in range(2):
                plt.subplot(height, width, 2 * i + j+1)
                if j%2 == 0:
                    plt.imshow(filters[i,0],cmap = 'gray',interpolation='none')
                else:
                    plt.imshow(torch.from_numpy(resulting_image), cmap='gray', interpolation='none')
                plt.xticks([])
                plt.yticks([])
        plt.show()

def loadCustomData(dataset_path, batch=10):
    '''
    Creates a sequential transformer and then loads the dataset to the dataloader given a path
    :param dataset_path: path which specifies the dataset to be loaded
    :return: the loader of the dataset
    '''
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.Grayscale(1),
        torchvision.transforms.RandomInvert(p=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.1307, 0.3081)])
    train_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=test_transform)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=False)
    return data_loader

def loadCustomDataModelCheck(dataset_path, batch=10):
    '''
    Creates a sequential transformer and then loads the dataset to the dataloader given a path
    :param dataset_path: path which specifies the dataset to be loaded
    :return: the loader of the dataset
    '''
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=test_transform)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=False)
    return data_loader

def sumSquareDist(val1, val2):
    '''
    Calculates the sum square distance between two vectors
    :param val1: data 1
    :param val2: data 2
    :return: returns the SSD
    '''
    ssd = 0
    for i in range(len(val1)):
        ssd += (val1[i].numpy() - val2[i].numpy()) * (val1[i].numpy() - val2[i].numpy())
    return ssd

def nearestNeighbour(ssd, i):
    '''
    Calculate the nerest neighbour
    :param ssd: values of all comparisions
    :param i: index of data
    :return: nearest neighbour index
    '''
    minV = None
    minVI = None
    for ind in range(ssd.itemsize):
        if minV == None or (ssd[ind] < minV and ind != i):
            minV = ssd[ind]
            minVI = ind
    return minVI

def sumSquareDistanceM(featureValues, i, newFeature = None, separateFeature = False ):
    '''
    From the list of features of each data element compare ith data with rest of the dataset
    :param featureValues: values containing features of all the images
    :param i: the element to compare the features with
    :return:
    '''
    minSSD = None
    ind = None
    feat = featureValues[i]
    if separateFeature == True:                                #searching in the same dataset
        feat = newFeature
        i = -1              #if new feature index should not matter
    for data in range(len(featureValues)):
        ssd = sumSquareDist(feat, featureValues[data])
        if i != data and (minSSD == None or ssd < minSSD):
            minSSD = ssd
            ind = data
    print(minSSD)
    return ind

def KNNclassifier(featureValues, feature):
    '''
    feature Values have 9 data for each feature
    :param featureValues:
    :param feature:
    :return:
    '''
    ind = None
    minSum = None
    sumV = 0
    for i in range(len(featureValues)):
        ssd = sumSquareDist(feature, featureValues[i])
        sumV += ssd
        if i%9 == 8:
            if minSum == None or sumV < minSum:
                minSum = sumV
                ind = (i-8)//9
            sumV = 0
    print(minSum)
    return ind

def drawClasOutput(output, strOut):
    '''
    Draws the result of how the ouput of truncated model task 2 looks like
    :param output: result from the truncated model
    :param strOut: the name of the string for the figure to pop up as
    :return: None
    '''
    plt.figure(strOut, figsize=(4, 5))
    for i in range(len(output)):
        plt.subplot(4, 5, i + 1)
        plt.imshow(output[i], cmap='gray', interpolation='none')
        plt.title(str(i + 1))
        plt.xticks([])
        plt.yticks([])
    plt.show()
