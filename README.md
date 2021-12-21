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

*Bit Width*
- AXWX means Quantization Bit-Width Set. (eg. AWBW : Activation X bit, Weight X bit Quantization)

*Clipping Value*
- Per Channel : Per channel means that for each dimension, typically the channel dimension of a tensor, the values in the tensor are scaled and offset by a different value
- Per Tensor(Per Layer) : Per tensor means that all the values within the tensor are scaled the same way. 

- PACT Activation Quantization's clipping initial value is constant. Our A2W2 MB1 PACT_LSQ model has its best performace when PACT activation Quantization's initial value was 3, Clipping Weight decay was 0.0005

*Quantization Method*
- Brevitas : Brevitas is a PyTorch research library for quantization-aware training (QAT). It supports FINN Compile
- PACT_SAWB : This means that Quantizing Activation uses PACT Algorithm, Quantizing Weight uses LSQ Algorithm.

*Quantization Alogorithm*
You can find more detailed information about Quantization in following link.
https://pytorch.org/docs/stable/quantization.html

- PACT : https://arxiv.org/abs/1805.06085 
- LSQ : https://arxiv.org/abs/1902.08153




## MobileNet V1

The reduced-precision implementation of MobileNet V1 makes the following assumptions:
- Floating point per-channel scale factors can be implemented by the target hardware, e.g. using FINN-style thresholds.
- Input preprocessing is modified to have a single scale factor rather than a per-channel one, so that it can be propagated through the first convolution to thresholds.
- Weights of the first layer are always quantized to 8 bit.
- Padding in the first convolution is removed, so that the input's mean can be propagated through the first convolution to thresholds.
- Scaling of the fully connected layer is per-layer, so that the output of the network doesn't require rescaling.
- Per-channel scale factors before depthwise convolution layers can be propagate through the convolution.
- Quantized avg pool performs a sum followed by a truncation to the specified bit-width (in place of a division).
