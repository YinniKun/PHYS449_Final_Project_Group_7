#main file for glycolosis

import json, argparse, torch, sys
import numpy as np
import torch.optim as optim
from src.nn_gen import Net
from src.data_gen import testdata

#all functions taken from Workshop 2... need to be fixed to fit the dataset(s) we are using -Callum
def prep_demo(n_training_data, n_test_data):
    """
    Prepares the model and dataset
    :return: model and data
    """

    model = Net(4)
    data = testdata(n_training_data, n_test_data)
    return model, data


def run_demo(param, model, data):

    # Define an optimizer and the loss function
    optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])
    loss = torch.nn.MSELoss(reduction= 'mean')

    obj_vals= []
    cross_vals= []
    num_epochs= int(param['num_epochs'])

    # Training loop
    for epoch in range(1, num_epochs + 1):

        train_val= model.backprop(data, loss, epoch, optimizer)
        obj_vals.append(train_val)

        test_val= model.test(data, loss, epoch)
        cross_vals.append(test_val)

        if (epoch+1) % param['display_epochs'] == 0:
            print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
                      '\tTraining Loss: {:.4f}'.format(train_val)+\
                      '\tTest Loss: {:.4f}'.format(test_val))


    print('Final training loss: {:.4f}'.format(obj_vals[-1]))
    print('Final test loss: {:.4f}'.format(cross_vals[-1]))

    return obj_vals, cross_vals

if __name__ == '__main__':
    model, data = prep_demo(10, 5)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss = torch.nn.MSELoss(reduction='mean')
    print(model.backprop_no_ode(data, loss, optimizer))
