# Author: A.M.Kharazi (adapted)
# License: BSD 3 clause

import torch
import torch.nn as nn
import torchvision.models as models


class _efficientnet_b4(nn.Module):
    """
    a classic EfficientNet-B4

    ----------
    pretrained : boolean
            if True, uses weights_path to pretrained the model
            if False, ignores the weights_path

    weights_path : str
            path to the .pth file

    tensorized : boolean
        if True, flattens the data before feeding it to the classifiers,
        if False, keeps the current structure of the data
    """

    def __init__(
        self,
        pretrained=True,
        weights_path="../weights/efficientnet_b4_weights.pth",
        tensorized=False,
    ):
        super(_efficientnet_b4, self).__init__()

        model = models.efficientnet_b4(weights=None)
        if pretrained:
            if weights_path is None:
                raise ValueError("weights_path cannot be None when pretrained=True")
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))

        self.tensorized = tensorized
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        if not self.tensorized:
            x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def out_shape_efficientnet_b4(in_shape=(380, 380), batch_size=1):
    """
    calculates the shape of tensor before classifier layers

    ----------
    in_shape : tuple
            input size of the image , use only the width and height

    batch_size : int
            a batch size index, default value is 1

    returns the shape of tensor before the classifier layers
    """
    dummy_input = torch.rand((batch_size, 3) + tuple(in_shape))
    model = _efficientnet_b4(pretrained=False, weights_path=None, tensorized=False)
    with torch.no_grad():
        dummy_out = model.features(dummy_input)
    return dummy_out.shape


def EfficientNetB4(
    pretrained=True,
    weights_path="../weights/efficientnet_b4_weights.pth",
    tensorized=False,
    input_shape=(380, 380),
    num_classes=1000,
    avg_pool=True,
    new_classifier=None,
):
    """
    creates a classic EfficientNet-B4

    ----------
    pretrained : boolean
            if True, uses weights_path to pretrained the model
            if False, ignores the weights_path

    weights_path : str
            path to the .pth file

    tensorized : boolean
            if True, flattens the tensor before classifier layers,
            if False, keeps the current structure of the data

    input_shape : tuple
            used to calculate the classifier in features

    avg_pool : boolean or nn.Module
            if True, performs the adaptive average pooling
            if False, does not perform the adaptive average pooling
            if nn.Module, replaces the adaptive average pooling layer of the model

    new_classifier : boolean or nn.Module or None
            if False, does not change the classifier layer of the model
            if nn.Module, replaces the classifier layers of the model

    returns an nn.Module model
    """
    model = _efficientnet_b4(
        pretrained=pretrained,
        weights_path=weights_path,
        tensorized=tensorized,
    )

    if isinstance(input_shape, list):
        input_shape = tuple(input_shape)

    if isinstance(avg_pool, bool):
        if not avg_pool:
            model.avgpool = nn.Identity()
    else:
        model.avgpool = avg_pool

    if new_classifier is not None:
        model.classifier = new_classifier
        return model

    if (input_shape != (380, 380) or num_classes != 1000) and (new_classifier is None):
        if isinstance(model.avgpool, nn.Identity):
            out_shape = out_shape_efficientnet_b4(in_shape=input_shape, batch_size=2)
            in_features = out_shape[1] * out_shape[2] * out_shape[3]
            p = model.classifier[0].p if hasattr(model.classifier[0], "p") else 0.4
            model.classifier = nn.Sequential(
                nn.Dropout(p=p, inplace=True),
                nn.Linear(in_features, num_classes),
            )
        else:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
