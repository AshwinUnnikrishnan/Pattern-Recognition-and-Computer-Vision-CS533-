# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:13:25 2022

@author: Gourav Beura

Extension 3:
    Replace the first layer of the MNIST network with a filter bank of your 
    choosing (e.g. Gabor filters) and retrain the rest of the network, holding 
    the first layer constant. How does it do?
"""

import cv2
from torch import nn
from torch import optim
import torch,torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import summary
from network import MyNetwork
import copy
import numpy 

#Custom CNN
class Submodel(MyNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the forward method
    def forward(self, x):
        x = F.sigmoid(F.max_pool2d(gabor(x), 2))
        x = F.sigmoid(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
def gabor(x):
    # print(x.shape)
    filters = generate_gabor_filters()
    
    batch_size = x.shape[0]
    samples = []
    for b in range(batch_size):
        channels = []
        for i in range(len(filters)):
            channel = cv2.filter2D(x[b][0].numpy(), -1, filters[i])
            channels.append(channel)
        samples.append(channels)

    # np.array(results)
    
    samples = numpy.array(samples)
    # print(samples.shape)
    return torch.tensor(samples)

def apply_gabor_filer_bank(image,filter_bank):
    """
    Returns a list of image frames convolved with gabor filters.

    Parameters
    ----------
    image : TYPE
        image matrix.
    filter_bank : TYPE
        list of gabor filters.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    conv_image_bank=[]
    for kernel in filter_bank:
        output = cv2.filter2D(image, -1, kernel)
        print("Filter2D: {}".format(output.shape))
        conv_image_bank.append(output)
    return numpy.array(conv_image_bank)

def get_gablor_filter_bank_layer(train_dl,test_dl,filter_bank):
    """
    
    Get a list of gabor filters
    
    Parameters
    ----------
    train_dl : TYPE
        DESCRIPTION.
    test_dl : TYPE
        DESCRIPTION.
    filter_bank : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    train_dl_modified=[]
    test_dl_modified=[]
    
    for batch_idx,(images,labels) in enumerate(train_dl):
        for image in images:
            image = image.cpu().numpy()
            print(image)
            image_gabor_bank = apply_gabor_filer_bank(image,filter_bank)
            arr = numpy.array(image_gabor_bank)
            print("Test Image: {}".format(arr.shape))
            train_dl_modified.append(arr)
    print("Test Data:")
    for batch_idx,(images,labels) in enumerate(test_dl):
        for image in images:
            image = image.cpu().numpy()
            print(image)
            image_gabor_bank = apply_gabor_filer_bank(image,filter_bank)
            test_dl_modified.append(image_gabor_bank)
    return numpy.array(train_dl_modified), numpy.array(test_dl_modified)

def get_gablor_filter_bank_layer_test(train_dl,filter_bank):
    """
    Testing gabor filter convolution

    Parameters
    ----------
    train_dl : TYPE
        DESCRIPTION.
    filter_bank : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    train_dl_modified=[]
    
    for batch_idx,(images,labels) in enumerate(train_dl):
        for image in images:
            image = image.cpu().numpy()
            print(image)
            image_gabor_bank = apply_gabor_filer_bank(image,filter_bank)
            arr = numpy.array(image_gabor_bank)
            print("Test Image: {}".format(arr.shape))
            train_dl_modified.append(arr)
    return numpy.array(train_dl_modified)


def generate_gabor_filters():
    """
    Parameters
    ----------
        
        Gabor Filter Bank Paramter Values:
            Lambda (λ): The wavelength governs the width of the strips of 
                    Gabor function. Eg.:30,60,100
            Theta (Ө): The theta controls the orientation of the Gabor function
                    Eg: (0,45,90)
            Gamma (ɣ): The aspect ratio or gamma controls the height of the 
                    Gabor function.
                    Eg: 0.25,0.5,0.75
            Sigma (σ): The bandwidth or sigma controls the overall size of the 
                    Gabor envelope.
                    Eg: 10,30,45
                    
            Possible Kernels:
                (Ө = 00, ɣ = 0.25, σ = 10, Ψ = 0) change lambda
                (λ = 30, ɣ = 0.25, σ = 10, Ψ = 0) change theta
                (λ = 30, Ө = 00, σ = 10, Ψ = 0) change gamma
                (λ = 30, Ө = 00 ɣ = 0.25, Ψ = 0) change sigma
            
    Returns
    -------
    None.

    """
    gabor_filter_bank=[]
    sigma_vals = [10,30,45]
    gamma_vals = [0.25,0.5,0.75]
    lambda_vals = [30,60,100]
    for i,sigma in enumerate(sigma_vals):
        if(i%2==0):
            gabor_filter_bank.append(cv2.getGaborKernel((5,5), sigma=sigma, theta=0, lambd=30, gamma=0.25))
        else:
            gabor_filter_bank.append(cv2.getGaborKernel((5,5), sigma=sigma, theta=180, lambd=30, gamma=0))
    
    for i,gamma in enumerate(gamma_vals):
        if(i%2==0):
            gabor_filter_bank.append(cv2.getGaborKernel((5,5), sigma=30, theta=90, lambd=60, gamma=gamma))
        else:
            gabor_filter_bank.append(cv2.getGaborKernel((5,5), sigma=45, theta=0, lambd=30, gamma=gamma))
            
    for i,l in enumerate(lambda_vals):
        if(i%2==0):
            gabor_filter_bank.append(cv2.getGaborKernel((5,5), sigma=10, theta=90, lambd=l, gamma=0.25))
        else:
            gabor_filter_bank.append(cv2.getGaborKernel((5,5), sigma=10, theta=135, lambd=l, gamma=0.25))
    
    gabor_filter_bank.append(cv2.getGaborKernel((5,5), sigma=10, theta=45, lambd=30, gamma=0.25))
    
    print(gabor_filter_bank)
    return gabor_filter_bank


def visualize_gabor_filters(gabor_filters):
    """
    
    Ploting the gabor Filters in a graph
    Parameters
    ----------
    gabor_filters : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #get the number of kernals
    num_kernels = len(gabor_filters)    
    
    #define number of columns for subplots
    num_cols = 5
    #rows = num of kernels
    num_rows = 2
    
    #set the figure size
    fig = plt.figure(figsize=(num_cols,num_rows))
    
    #looping through all the kernels
    for i in range(num_kernels):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(gabor_filters[i],cmap='gray')
        ax1.axis('off')
        ax1.set_title(str(i+1))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
           
    plt.tight_layout()
    plt.show()
    
    

def modify_model_layer(model,gabor_filters):
    """
    Generate the summary of mopdel

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    gabor_filters : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    summary(model,(1,28,28))
    pass

def plot_filter_image(filter_bank,data):
    """
    
    The the image having gabor filters
    Parameters
    ----------
    filter_bank : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    data, target = next(iter(data))
    
    data = data[0].numpy()
    data = data.transpose(1,2,0)
    
    effects=[]
    for kernel in gabor_filters:
        effects.append(cv2.filter2D(data, -1, kernel[0]))
    
    k=1
    fig = plt.figure(figsize=(4,5))
    for i in range(10):
        plt.subplot(5,4,k)
        plt.imshow(gabor_filters[i], cmap='gray',interpolation='none')
        k+=1
        plt.xticks([])
        plt.yticks([])
    
        plt.subplot(5,4,k)
        plt.imshow(effects[i], cmap='gray', interpolation='none')
        k+=1
        plt.xticks([])
        plt.yticks([])
    fig.show()


if __name__=="__main__":
    
    #Hyper-parameters
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    
    #step 1: generate filter bank
    gabor_filters = generate_gabor_filters()
    visualize_gabor_filters(gabor_filters)
