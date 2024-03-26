import torch
import torch.nn as nn
import torchvision.models as models

def out_shape_vgg19(in_shape = (224,224), batch_size = 1):
    dummy_input = torch.rand((batch_size, 3) + in_shape)
    model = models.vgg19(weights=None)
    x = model.features(dummy_input)
    # output shape before avg pool layer and classifier layers
    return x.shape
  
def VGG19(pretrained  = True,
          weights_path = '../weights/vgg19_weights.pth',
          input_shape = (224,224),
          num_classes = 1000,
          avg_pool = True,
          new_classifier = None):
    
    model = models.vgg19(weights=None)
    if pretrained:
        model.load_state_dict(torch.load(weights_path))
        
    if isinstance(input_shape, list):
        input_shape = tuple(input_shape)      
    if isinstance(avg_pool, bool):
        if not avg_pool:
            model.avgpool = nn.Identity()
    else:
        model.avgpool = avg_pool
        
    if (input_shape != (224,224) or num_classes != 1000) and (new_classifier is None or new_classifier is False):
        out_shape = out_shape_vgg19(in_shape=input_shape, batch_size = 5)
        model.classifier[0] = nn.Linear(out_shape[1] * out_shape[2] * out_shape[3], 4096)
        model.classifier[6] = nn.Linear(4096, num_classes)

    if new_classifier is not None:
        model.classifier = new_classifier
    
    return model