# Author: A.M.Kharazi
# License: BSD 3 clause

import torch
import torch.nn as nn
import torchvision.models as models

class _resnet101(nn.Module):
    '''
    a classic ResNet101

    ----------
    pretrained : boolean 
            if True, uses weights_path to pretrained the model
            if False, ignores the weights_path

    weights_path :  str
            path to the .pth file

    tensorized : boolean
        if True, flattens the data before feeding it to the classifiers,
        if False, keeps the current structure of the data
    '''
    def __init__(self, pretrained = True, weights_path = '../weights/resnet101_weights.pth', tensorized = False):
        super(_resnet101, self).__init__()
        
        model = models.resnet101(weights= None)
        if pretrained:
           model.load_state_dict(torch.load(weights_path))
        
        self.tensorized = tensorized
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        
        self.avgpool = model.avgpool
        
        self.classifier = nn.Sequential(
            model.fc
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        if not self.tensorized:
           x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def out_shape_resnet101(in_shape = (224,224), batch_size = 1):
    '''
    calculates the shape of tensor before classifier layers

    ----------
    in_shape : tuple 
            input size of the image , use only the width and height
    
    batch_size : int
            a batch size index, default value is 1

    
    returns the shape of tensor before the classifier layers
    '''
    dummy_input = torch.rand((batch_size, 3) + in_shape)
    model = _resnet101(pretrained=False, weights_path=None, tensorized=False)
    dummy_input = model.features(dummy_input)
    # output shape before avg pool layer and classifier layers
    return dummy_input.shape
  
def Resnet101(pretrained  = True,
              weights_path = '../weights/resnet101_weights.pth',
              tensorized = False,
              input_shape = (224,224),
              num_classes = 1000,
              avg_pool = True,
              new_classifier = None):
    '''
    creates a classic ResNet101

    ----------
    pretrained : boolean 
            if True, uses weights_path to pretrained the model
            if False, ignores the weights_path

    weights_path :  str
            path to the .pth file
    
    tensorized : boolean
            if True, flattens the tensor before classifier layers,
            if False, keeps the current structure of the data

    input_shape : tuple
            used to calculate the classifier in features
    
    avg_pool : boolean or nn.Module
            if True, does not perform a adaptive average pooling
            if False, performs the adaptive average pooling 
            if nn.Module, replaces the adaptive average pooling layer of the model
 
    new_classifier : boolean or nn.Module or None
            if False, does not change the classifier layer of the model 
            if nn.Module, replaces the classifier layers of the model

    
    returns an nn.Module model
    '''
    
    model = _resnet101(pretrained= pretrained,
                        weights_path= weights_path,
                        tensorized= tensorized)
        
    if isinstance(input_shape, list):
        input_shape = tuple(input_shape)      
    if isinstance(avg_pool, bool):
        if not avg_pool:
            model.avgpool = nn.Identity()
    else:
        model.avgpool = avg_pool
        
    if (input_shape != (224,224) or num_classes != 1000) and (new_classifier is None):
        out_shape = out_shape_resnet101(in_shape=input_shape, batch_size = 5)
        model.classifier[0] = nn.Linear(out_shape[1] * out_shape[2] * out_shape[3], num_classes)
        
    if new_classifier is not None:
        model.classifier = new_classifier
    
    return model