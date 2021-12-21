import os
from configparser import ConfigParser
import torch
from torch import hub

from .mobilenetv1 import *
from .mobilenetv1_CIFAR10 import *
from .mobilenetv1_FP_CIFAR10 import *
from .vgg import *
from .proxylessnas import *

model_impl = {
    'quant_mobilenet_v1': quant_mobilenet_v1,
    'quant_proxylessnas_mobile14': quant_proxylessnas_mobile14,
    'quant_mobilenet_v1_cifar10_2b' : quant_mobilenet_v1_cifar10,
    'quant_mobilenet_v1_FP_cifar10' : quant_mobilenet_v1_FP_cifar10
}


def model_with_cfg(name, pretrained, ft_dir=None):
    cfg = ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'cfg', name + '.ini')
    assert os.path.exists(config_path)
    cfg.read(config_path)
    arch = cfg.get('MODEL', 'ARCH')
    
    model = model_impl[arch](cfg)
    if pretrained:
        if ft_dir is not None:
            checkpoint_dir = ft_dir
        else:
            checkpoint_dir = cfg.get('MODEL', 'PRETRAINED_DIR')
        #state_dict = hub.load_state_dict_from_url(checkpoint, progress=True, map_location='cpu')
        state_dict = torch.load(checkpoint_dir)

        weight_mean = dict()
        model_dict = dict()

        if ft_dir is None:    
            for n, p in state_dict['state_dict'].items():
                if "conv.weight" in n and not "weight_quant.tensor_quant.s" in n:
                    model_dict[n] = p
                    mean = p.abs().mean() * 2
                    #mean = (p.std() * 3) / 2 LSQ PLUS
                    model_dict[n+'_quant.tensor_quant.s'] = mean.view(1)
                elif "weight_quant.tensor_quant.s" in n or "features.init_block.conv.weight_quant.tensor_quant.s" in n:
                    continue
                else:
                    model_dict[n] = p
        else:
            model_dict = state_dict['state_dict']
        print(f"====> Load Pretrained Model - ACC : {state_dict['best_val_acc']}")
        print(model.load_state_dict(model_dict, strict=False))
        #model.load_state_dict(state_dict['state_dict'], strict=False)
    return model, cfg


def quant_mobilenet_v1_4b(pretrained=True):
    model, _ = model_with_cfg('quant_mobilenet_v1_4b', pretrained)
    return model

def quant_mobilenet_v1_cifar10(pretrained=False):
    model, _ = model_with_cfg('quant_mobilenet_v1_cifar10', pretrained)
    return model

def quant_mobilenet_v1_FP_cifar10(pretrained=False):
    model, _ = model_with_cfg('quant_mobilenet_v1_FP_cifar10_2b', pretrained)
    return model


def quant_proxylessnas_mobile14_4b(pretrained=True):
    model, _ = model_with_cfg('quant_proxylessnas_mobile14_4b', pretrained)
    return model


def quant_proxylessnas_mobile14_4b5b(pretrained=True):
    model, _ = model_with_cfg('quant_proxylessnas_mobile14_4b5b', pretrained)
    return model


def quant_proxylessnas_mobile14_hadamard_4b(pretrained=True):
    model, _ = model_with_cfg('quant_proxylessnas_mobile14_hadamard_4b', pretrained)
    return model