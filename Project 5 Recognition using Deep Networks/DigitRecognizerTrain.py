# Ashwin Unnikrishnan
# Project 5: Recognition using Deep Networks
# Date : 7 April 2022
# main file


import torch
import sys
import torch.optim as optim
import lib as library
from network import MyNetwork


random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


def main(argv):
    batch_test = 1000
    batch_train = 64
    learning_rate = 0.01
    momentum = 0.5
    n_epochs = 5
    log_interval = 10

    train_loader, test_loader = library.testLoader('mnist', batch_test, batch_train)

    print("Examining Test Data")
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)

    #Do not delete
    #Drawing the dataset
    library.plotGraph(6,example_data, example_targets, "Six Objects")

    #Drawing the network
    #Do not delete
    library.visualizeNetwork(MyNetwork(), example_data)


    #training the dataset
    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr= learning_rate, momentum=momentum)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    test_losses = library.test(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
        train_losses, train_counter = library.train(epoch, network, optimizer, log_interval, train_loader, 'MNIST',
                                                    train_losses, train_counter)
        test_losses = library.test(network, test_loader, test_losses)


    #Evaluating performance
    library.evalPerformance(train_counter, train_losses, test_counter, test_losses)


if __name__ == "__main__":
    main(sys.argv)