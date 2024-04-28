import torch
import torch.nn as nn

class _ConvolutionalNN(nn.Module):
    def __init__(self, tensorized = False):
        super(_ConvolutionalNN, self).__init__()
        self.tensorized = tensorized
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avgpool = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5, inplace=False),
            nn.Identity(),
            nn.Linear(1024, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        if not self.tensorized:
           x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def out_shape_cnn(in_shape = (224,224), batch_size = 1):
    dummy_input = torch.rand((batch_size, 3) + in_shape)
    model = _ConvolutionalNN(tensorized=False)
    dummy_input = model.features(dummy_input)
    # output shape before avg pool layer and classifier layers
    return dummy_input.shape

def CNN(pretrained  = False,
        weights_path = '../weights/cnn_weights.pth',
        tensorized = False,
        input_shape = (224,224),
        num_classes = 1000,
        avg_pool = True,
        new_classifier = None):
    
    model = _ConvolutionalNN(tensorized=tensorized)
    if pretrained:
        model.load_state_dict(torch.load(weights_path))
        
    if isinstance(input_shape, list):
        input_shape = tuple(input_shape)      
    if isinstance(avg_pool, bool):
        if not avg_pool:
            model.avgpool = nn.Identity()
    else:
        model.avgpool = avg_pool
        
    if (input_shape != (224,224) or num_classes != 1000) and (new_classifier is None):
        out_shape = out_shape_cnn(in_shape=input_shape, batch_size = 5)
        model.classifier[0] = nn.Linear(out_shape[1] * out_shape[2] * out_shape[3], 1024)
        model.classifier[3] = nn.Linear(1024, num_classes)

    if new_classifier is not None:
        model.classifier = new_classifier
    
    return model