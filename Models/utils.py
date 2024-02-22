import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import os
'''
Contains functions for :
1. Train/Test Loader for Cifar10
2. Train/Test Loader for MNIST
3. Count the parameter of a model
4. Topk Accuracy calculation
'''


def load_cifar(BATCH_SIZE=16, PATH='./data'):
    '''
    Makes train and test loader for CIFAR10 dataset
    BATH_SIZE : Is the size of batches - Type : Int
    PATH : Is the path where the dataset is going to be downloaded or placed
    '''
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(192),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(192),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=PATH, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=PATH, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #        'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader


def load_mnist(BATCH_SIZE=16, PATH='./data'):
    '''
    Makes train and test loader for MNIST dataset
    BATH_SIZE : Is the size of batches - Type : Int
    PATH : Is the path where the dataset is going to be downloaded or placed
    To make the MNIST dataset applicable for similar models as other datasets, 
    we transformed 1 channel images into 3 channel images 
    '''
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(192),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307),
                             (0.3081, 0.3081, 0.3081)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(192),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307),
                             (0.3081, 0.3081, 0.3081)),
    ])
    trainset = torchvision.datasets.MNIST(
        root=PATH, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(
        root=PATH, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return trainloader, testloader


def count_param(model):
    '''
    Counts the number of parameters in a model
    model : nn.Module 
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def topk(output, target, k):
    '''
    Returns the TopK accuracy of a model
    output : Is the predicted values - Type : Tesnsor
    target : Is the actual values - Type : Tensor
    k : Is the TopK'th accuracy 

    '''
    correct = 0.0
    batch_size = output.size(0)
    _, topk_indices = output.topk(k, dim=1, largest=True, sorted=True)
    target = target.view(-1, 1).expand_as(topk_indices)

    correct += (topk_indices == target).sum().item()

    return (correct / batch_size) * 100.0
