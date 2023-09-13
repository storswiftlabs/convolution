import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np
import sys
import os
import math



class ConvNet(nn.Module):
    # Initializing the network structure: defining the layers of the network
    def __init__(self):
        super(ConvNet, self).__init__()

        # Sequential method to create ordered layers
        # Conv2d method of nn.Module, creates a set of convolution filters
        # First argument is the number of input channels, second argument is the number of output channels
        # For convolution filter of x * y, the parameter is a tuple (x, y)
        self.layer1 = nn.Sequential(
            # Dimensional change during convolution and pooling operation formula: width_of_output = (width_of_input - filter_size + 2 * padding) / stride + 1
            nn.Conv2d(1, 5, kernel_size=3, stride=1, padding=1),
            # Activation function
            nn.ReLU(),
            # kernel_size: pooling size, stride: down-sample
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # The purpose of connecting these layers is to add the rich information output by the neural network to the standard classifier
        self.fc1 = nn.Linear(490, 10)
        self.fc2 = nn.Linear(10, 10)

    # Define the forward propagation of the network, this function will override the forward function in nn.Module
    # Input x goes through the layers of the network, and the output is out
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # Flattens the data dimensions from 7 x 7 x 64 into 3164 x 1
        # Fixed the side with -1 to have only one column
        out = out.reshape(out.size(0), -1)
        # Drop out some neural units with a certain probability to prevent overfitting
        # out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def array_int2str(array):
    for i in range(len(array)):
        array[i] = str(array[i])
    return str(json.dumps(array))

def generate_prover_file(output_path, model_store_path, test_dataset_path):

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = torchvision.datasets.MNIST(root=test_dataset_path, train=True, transform=trans, download=True)
    test_dataset = torchvision.datasets.MNIST(root=test_dataset_path, train=False, transform=trans)

    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = ConvNet()
    model.load_state_dict(torch.load(model_store_path))  # Loading model weights


    inputs = {}

    for i, key in enumerate(model.state_dict().keys()):
        layer_num = int(i / 2)+1
        value = model.state_dict()[key]
        shape = value.shape
        value_list = value.view(-1).numpy().tolist()
        if 'weight' in key:
            name = "w_" + str(layer_num)
        if 'bias' in key:
            name = "b_" + str(layer_num)
        inputs[name]=value_list
        print(len(inputs[name]))

    for images in test_loader:
        name = 'inputs'
        inputs[name] = images[0].view(-1).numpy().tolist()
        break

    scale = 2**32
    _zero_point = 2**64
    for k in inputs:
        v = inputs[k]
        v = [int(a * scale + _zero_point) for a in v]
        inputs[k] = v
    prover_str = '\n'.join([k + ' = ' + array_int2str(inputs[k]) for k in inputs])

    with open(os.path.join(output_path, "Prover.toml"), "w+") as file:
        file.write(prover_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loading model parameters and testing data generates Noir code.")
    parser.add_argument("-o", "--output_path", type=str, required=True,
                        help="Save the generated 'main.nr' and 'Prover.toml' to the specified path.")
    parser.add_argument("-m", "--model_store_path", type=str, required=True,
                        help="The location of the trained model's path.")
    parser.add_argument("-t", "--test_dataset_path", type=str, required=True,
                        help="Dataset for testing.")
    args = parser.parse_args()
    generate_prover_file(args.output_path, args.model_store_path, args.test_dataset_path)
    # generate_prover_file('/mnt/code/convolution/examples/cnn', '/mnt/code/convolution/static/modelconv_net_model.ckpt', '/mnt/code/convolution/static/dataset')
