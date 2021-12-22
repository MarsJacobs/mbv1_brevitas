# CIFAR10 MB1 Brevitas Quantization Training Framework

This Framework supports CIFAR10 MobileNetV1(MB1) PACT, LSQ Quantization aware Training based on FINN Brevitas.
Below in the table is a list of CIFAR10 Quantized MB1 Performances. 

Clip Value | Quantization Method  | A4W4 | A2W2
-- | -- | -- | --
Per Channel | Brevitas | 91.53 | -
Per Layer | Brevitas | 91.29 | -
Per Layer | PACT_SAWB | 91.37 | 85.78
Per Layer | PACT_LSQ | 91.44 | 87.42

### Quantization Setting

Before Quantization Aware Trainig, We initialized model weight using Full Preicision MB1 model file. 

CIFAR10 Full Precision MB1 Accuracy is **91.6**.  
(We trained Full Precision MB1 model using this code. you can use your own Full Precision MB1 Model)

**Bit Width**
- AXWX means Quantization Bit-Width Set. (eg. AWBW : Activation X bit, Weight X bit Quantization)

**Clipping Value**
- Per Channel : Per channel means that for each dimension, typically the channel dimension of a tensor, the values in the tensor are scaled and offset by a different value
- Per Tensor(Per Layer) : Per tensor means that all the values within the tensor are scaled the same way. 

- PACT Activation Quantization's clipping initial value is constant. Our A2W2 MB1 PACT_LSQ model has its best performace when PACT activation Quantization's initial value was 3, Clipping Weight decay was 0.0005

**Quantization Method**
- Brevitas : Brevitas is a PyTorch research library for quantization-aware training (QAT). It supports FINN Compile
- PACT_SAWB : This means that Quantizing Activation uses PACT Algorithm, Quantizing Weight uses LSQ Algorithm.

**Quantization Alogorithm**
You can find more detailed information about Quantization in this link.
- Quantization : https://pytorch.org/docs/stable/quantization.html
- PACT : https://arxiv.org/abs/1805.06085 
- LSQ : https://arxiv.org/abs/1902.08153

## Traininig
- You can try Quantization Aware Training with this command.
- Your Pretrained Full Precision Model file's directory shoulb be specified in cfg/quant_mobilenet_v1_cifar10_2b.ini, PRETRAINED_DIR

      python imagenet_train.py --network quant_mobilenet_v1_cifar10_2b --experiments ./experiments --optim SGD --scheduler STEP --pretrained --gpus 0 --lr 0.01 --weight_decay 0.003 

## Accelerator Deploy Results

After Training QNN using Brevitas, you can export onnx file for finn compiler (refer to onnx_make.py)
https://github.com/xilinx/finn
Using finn compiler, you can make bitfile for 2bit Quantized CIFAR10 MobileNetv1 Inference on Alveo FPGA Board.

FPGA Board Inference time and Hardware Usage results are followed.

**Hardware Unit Usage**
  | PACT A4W4 | PACT A2W2
-- | -- | --
Date | 20211216 | 20211220
Dataset | CIFAR10 | CIFAR10
Total LUTs | 514905 | 275234
LUTRAM | 24269 | 17495
Flip-Flop | 564244 | 374026
BRAM36 | 703 | 490
BRAM18 | 135 | 41
URAM | 32 | 22
DSP | 103 | 103

**FPGA Board Inference Performance**
  | Date | Dataset | Pytorch ACC | HW Acc | Full Validation Time (per loop) (ms) | Runtime (ms) | Throughput [images/s] | DRAM in BW | DRAM out BW | fclk (mhz) | batch_size | fold_input (ms) | pack_input (ms) | copy_input_data_to_device (ms) | copy_output_data_to_device (ms) | unpack_output (ms) | unfold_output (ms)
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
PACT A4W4 | 20211216 | CIFAR10 | 91.44 | 91.34 | 535 | 51.72 | 19333.93 | 59.39 | 0.02 | 245 | 1000 | 0.04 | 0.026 | 1.21 | 0.167 | 0.31 | 0.014
PACT A2W2 | 20211220 | CIFAR10 | 87.40 | 87.45 | 635.00 | 61.85 | 16,169.31 | 49.67 | 0.02 | 209.00 | 1,000.00 | 0.02 | 0.02 | 1.26 | 0.13 | 0.23 | 0.03
