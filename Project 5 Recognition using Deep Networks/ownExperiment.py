# Ashwin Unnikrishnan
# Project 5: Recognition using Deep Networks
# Date : 7 April 2022
# task 4


import torch
import sys
import torch.optim as optim
import lib as library
from network import MyNetwork
import csv


random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


def main(argv):
    batch_test = 1000
    batch_train = [32,40]            #[1,32,64,128]
    learning_rate = 0.01
    momentum = 0.5
    n_epoch = [5,8,10]
    log_interval = 10
    f = open('csv_file_1', 'w')
    writer = csv.writer(f)


    for n_epochs in n_epoch:
        for batc in batch_train:
            train_loader, test_loader = library.testLoader('mnist', batch_test, batc)

            #print("Examining Test Data")
            examples = enumerate(test_loader)
            batch_idx, (example_data, example_targets) = next(examples)
            #print(example_data.shape)

            #training the dataset
            for p in range(1,11,1):
                tempRow = []
                tempRow.append(n_epochs)
                tempRow.append(batc)
                pp = p/float(10)
                tempRow.append(pp)
                print("Starting the run with epochs as {0} and batchsize as {1} and dropout as {2}".format(n_epochs, batc, pp))
                network = MyNetwork(pp)
                optimizer = optim.SGD(network.parameters(), lr= learning_rate, momentum=momentum)
                train_losses = []
                train_counter = []
                test_losses = []
                test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

                test_losses, predAcc = library.test(network, test_loader, test_losses, 'accuracy')
                for epoch in range(1, n_epochs + 1):
                    train_losses, train_counter = library.train(epoch, network, optimizer, log_interval, train_loader, 'MNIST',
                                                                train_losses, train_counter)
                    test_losses, predAcc = library.test(network, test_loader, test_losses,  'accuracy')

                print("Prediction Accuracy = {0}".format(predAcc))
                #Evaluating performance
                tempRow.append((predAcc.numpy()))
                writer.writerow(tempRow)

                library.evalPerformance(train_counter, train_losses, test_counter, test_losses, '_Accuracy_'+ str(predAcc) +'_nepochs' + str(n_epochs) + '_bat' + str(batc) + '_dropout' + str(pp))
    f.close()


if __name__ == "__main__":
    main(sys.argv)