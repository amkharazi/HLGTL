'''
Download ResNet50, ResNe101, and VGG19 Weights and store them in ./weights by default
'''

# Author: A.M.Kharazi
# License: BSD 3 clause

import os
import torch
from torchvision import models
import argparse


def download_resnet50_weights(save_dir):
    '''
    download ResNet50 weights, stored at save_dir

    ----------
    save_dir : str
        stored location
    '''
    print('Downloading ResNet50 weights ... ')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'resnet50_weights.pth')
    if not os.path.exists(save_path):
        resnet50 = models.resnet50(weights='DEFAULT')
        torch.save(resnet50.state_dict(), save_path)
    print('Resnet50 weights downloaded successfully.')
    
def download_resnet101_weights(save_dir):
    '''
    download ResNet101 weights, stored at save_dir

    ----------
    save_dir : str
        stored location
    '''
    print('Downloading ResNet101 weights ... ')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'resnet101_weights.pth')
    if not os.path.exists(save_path):
        resnet101 = models.resnet101(weights='DEFAULT')
        torch.save(resnet101.state_dict(), save_path)
    print('ResNet101 weights downloaded successfully.')

def download_vgg19_weights(save_dir):
    '''
    download VGG19 weights, stored at save_dir

    ----------
    save_dir : str
        stored location
    '''
    print('Downloading VGG19 weights ... ')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'vgg19_weights.pth')
    if not os.path.exists(save_path):
        vgg19 = models.vgg19(weights='DEFAULT')
        torch.save(vgg19.state_dict(), save_path)
    print('VGG19 weights downloaded successfully.')

def main(model, save_dir):
    '''
    download weights, stored at save_dir

    ----------
    model : str
        choices = ['resnet50', 'resnet101', 'vgg19', None]
        if None, then every model weights will be downloaded
        
    save_dir : str
        stored location
    '''
    if save_dir is None:
        save_dir = './weights'
    if model is None:
        download_resnet50_weights(save_dir)
        download_resnet101_weights(save_dir)
        download_vgg19_weights(save_dir)
    elif model == 'resnet50':
        download_resnet50_weights(save_dir)
    elif model == 'resnet101':
        download_resnet101_weights(save_dir)
    elif model == 'vgg19':
        download_vgg19_weights(save_dir)
    else:
        print('Invalid model choice. Please choose from vgg19, resnet50, or resnet101.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download and save pretrained model weights.')
    parser.add_argument('--model', choices=['resnet50', 'resnet101', 'vgg19'],
                        default= None, help='Choose model ( vgg19, resnet50, or resnet101 ) to download weights for. If None, then the weights for all models will be downloaded')
    parser.add_argument('--save_dir', default=None, help='Path to the folder to save the weights. If None, then the path is set to ./weights')
    args = parser.parse_args()

    main(args.model, args.save_dir)
