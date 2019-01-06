from .resnet import *
from .alexnet import *
from .vgg import *
from .squeezenet import *
from .densenet import *
from .inception import *

model_dicts = [resnet_dict, alex_dict, vgg_dict, squeezenet_dict, densenet_dict, inception_dict]
model_names = []
for model_dict in model_dicts:
    model_names += list(model_dict)

def initialize_model(model_name, classify_layers, pretrained=True):
    assert model_name in model_names, f"model name Error, {model_name} not in {model_names}"

    kwargs = {
        'model_name': model_name,
        'classify_layers': classify_layers,
        'pretrained': pretrained,
    }
    model, size = None, None
    if model_name.startswith('resnet'):
        model, size = initialize_resnet(**kwargs)
    elif model_name.startswith('alexnet'):
        model, size = initialize_alex(**kwargs)
    elif model_name.startswith('vgg'):
        model, size = initialize_vgg(**kwargs)
    elif model_name.startswith('squeezenet'):
        model, size = initialize_squeezenet(**kwargs)
    elif model_name.startswith('densenet'):
        model, size = initialize_densenet(**kwargs)
    elif model_name.startswith('inception'):
        model, size = initialize_inception(**kwargs)
    if hasattr(size, '__len__'):
        size = size[0]
    model.name = model_name
    return model, size