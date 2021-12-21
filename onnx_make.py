import argparse
import os
import random
import configparser

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models import *

import brevitas.onnx as bo
from brevitas.export import StdQOpONNXManager
SEED = 123456


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='quant_mobilenet_v1_cifar10_2b', help='Name of the model')
    parser.add_argument('--model_dir', type=str, default='', help='Name of the model')
    parser.add_argument('--output_dir', type=str, default='mobilenetv1_4b_baseline.onnx', help='Name of the output onnx name')
    
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    model, cfg = model_with_cfg(args.model, True, args.model_dir)
    print("==> Model Config DONE")
    
    bo.export_finn_onnx(model, (1, 3, 32, 32), args.output_dir)
    print("==> EXPORT DONE")
    #FINNManager.export(model, input_shape=(1, 3, 224, 224), export_path='mobilenetv1_pact.onnx')    

if __name__ == '__main__':
    main()
