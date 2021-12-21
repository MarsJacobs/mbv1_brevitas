from brevitas.inject import ExtendedInjector
from brevitas.inject import value
from brevitas.core.scaling import ParameterScaling

from brevitas.proxy import WeightQuantProxyFromInjector, ActQuantProxyFromInjector
# BaseInjector = _ExtendedInjectorType(
#     "Injector",
#     (),
#     {"__init__": __init__, "__doc__": injector_doc, "let": classmethod(let)})

from brevitas.quant.scaled_int import Int8ActPerTensorFloat

from brevitas.core.restrict_val import RestrictValueType
from brevitas.quant import Uint8ActPerTensorFloatMaxInit, Int8ActPerTensorFloatMinMaxInit
from brevitas.quant import Int8WeightPerTensorFloat
from .pact_func import *
import torch.tensor as Tensor

class CommonIntWeightPerTensorQuant(Int8WeightPerTensorFloat):
    """
    Common per-tensor weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None


class CommonIntWeightPerChannelQuant(CommonIntWeightPerTensorQuant):
    """
    Common per-channel weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_per_output_channel = False


class CommonIntActQuant(Int8ActPerTensorFloatMinMaxInit):
    """
    Common signed act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    min_val = -10.0
    max_val = 10.0
    restrict_scaling_type = RestrictValueType.LOG_FP


class CommonUintActQuant(Uint8ActPerTensorFloatMaxInit):
    """
    Common unsigned act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    
    scaling_min_val = 2e-16
    bit_width = 8
    max_val = 6.0
    restrict_scaling_type = RestrictValueType.LOG_FP

class CommonInputQuant(Int8WeightPerTensorFloat):
    proxy_class = ActQuantProxyFromInjector
    bit_width=8
    signed=True

# ================================================================================  #
# PACT Quantization Implementation
# ================================================================================ #

class LSQ_Quantizer(torch.nn.Module):
    def __init__(self, bit_width, is_activation=False):
        super(LSQ_Quantizer, self).__init__()
        
        self.bit_width = bit_width
        
        if(is_activation):
            self.Qn = 0
            self.Qp = 2 ** bit_width - 1
        else:
            self.Qn = -2**(bit_width - 1)
            self.Qp = 2 ** (bit_width - 1) - 1
        
        self.s = torch.nn.Parameter(torch.ones(1))

    def grad_scale(self, x, scale):
        y_out = x
        y_grad = x * scale

        y = (y_out - y_grad).detach() + y_grad

        return y

    def round_pass(self, x):
        y_out = x.round()
        y_grad = x
        y = torch.detach(y_out - y_grad) + y_grad

        return y

        # self.quantize_weight = LearnedTwosidedClippedLinearQuantization( num_bits = self.num_bits,
        #                                                                  init_clip_val = self.clip_init_val, 
        #                                                                  init_clip_valn = self.clip_init_valn,
        #                                                                  dequantize = True, 
        #                                                                  inplace = False) 

    def forward(self, x: Tensor) -> (Tensor, Tensor, Tensor):
        
        scale_factor = torch.Tensor([1 / (x.numel() * self.Qp) ** 0.5]).to(x)
        zero_point = torch.Tensor([0]).to(x)
        bit_width = torch.Tensor([self.bit_width]).to(x)
        
        scale = self.grad_scale(self.s, scale_factor)
        
        x = x / scale
        x = x.clamp(self.Qn, self.Qp)

        x_bar = self.round_pass(x)

        x_hat = x_bar * scale
        return x_hat, scale, zero_point, bit_width
        
class PACT_Weight_Quantizer(torch.nn.Module):
    def __init__(self, bit_width, is_activation=False):
        super(PACT_Weight_Quantizer, self).__init__()
        
        self.bit_width = bit_width
        #self.s = torch.nn.Parameter(torch.ones(1))

        self.scale = (2 **self.bit_width - 1)
        self.zero_point = 0.0
        self.clip_val = None

    def sawb_quantize_param(self, out, num_bits): # out is weight
        
        #scale, zero_point = asymmetric_linear_quantization_params(num_bits, 0, 1, signed=False)
        self.clip_val = self.sawb_quantization_params(self.bit_width, out)
        out = out.mul(1/self.clip_val).clamp(-1, 1).mul(0.5).add(0.5)
        #out = self.linear_quantize(out, self.scale, self.zero_point)
        out = LinearQuantizeSTE.apply(out, self.scale, self.zero_point, True, False)
        out = (2 * out - 1) * self.clip_val
        return out
    
    def sawb_quantization_params(self, num_bits, out):
        with torch.no_grad():
            x = out.flatten()
            mu = x.abs().mean()
            std = x.mul(x).mean().sqrt()

            dic_coeff = {2:(3.12, -2.064), 3:(7.509, -6.892), 4:(12.68, -12.80), 5:(17.74, -18.64)}
            coeff = dic_coeff[num_bits]
            clip_val = coeff[1] * mu + coeff[0] * std

            return clip_val

    def clamp(self, input, min, max, inplace=False):
        if inplace:
            input.clamp_(min, max)
            return input
        return torch.clamp(input, min, max)

    def minmax_quantize_param(self, out, num_bits): # out is weight
        
        #scale, zero_point = asymmetric_linear_quantization_params(num_bits, 0, 1, signed=False)
        clip_val = out.max()
        out = out.mul(1/clip_val).clamp(-1, 1).mul(0.5).add(0.5)
        #out = self.linear_quantize(out, self.scale, self.zero_point)
        out = LinearQuantizeSTE.apply(out, self.scale, self.zero_point, True, False)
        out = (2 * out - 1) * clip_val
        return out
    

    def forward(self, x: Tensor) -> (Tensor, Tensor, Tensor):
        
        if self.bit_width > 2:

            init_clip_val = x.max()
            self.clip_val = init_clip_val
            # init_clip_valn = x.min()
            # n = 2 ** self.bit_width -1 
            output = self.clamp(x, init_clip_val.item()*-1, init_clip_val.item())
            qweight = self.minmax_quantize_param(x, self.bit_width)
            


        else:
            qweight = self.sawb_quantize_param(x, self.bit_width)
        
        scale = ((self.clip_val * 2 / (2**self.bit_width -1)) / 2).to(x)
        zero_point = torch.Tensor(torch.zeros(1)).to(x)    
        # scale.to(x)
        # zero_point.to(x)

        bit_width = torch.Tensor([self.bit_width]).to(x)
        
        return qweight, scale, zero_point, bit_width
        # init_clip_val = self.s
        # init_clip_valn = self.s * -1
        # n = 2 ** self.bit_width -1 

        # diff = init_clip_val - init_clip_valn
        # scale = n / diff
        # zero_point = scale * init_clip_valn

        # scale.to(x)
        # zero_point.to(x)
        # bit_width = torch.Tensor([self.bit_width]).to(x)
        
        # output = self.clamp(x, init_clip_valn.item(), init_clip_val.item())
        # output = self.linear_quantize(output, scale, zero_point)
        # output = self.linear_dequantize(output, scale, zero_point)
        
        # return output, 1 / scale,  -1 *zero_point, bit_width

class PACT_ACT_Quantizer(torch.nn.Module):
    def __init__(self, bit_width, is_activation=False):
        super(PACT_ACT_Quantizer, self).__init__()
        
        self.bit_width = bit_width
        self.s = torch.nn.Parameter(torch.ones(1)*3)

    def clamp(self, input, min, max, inplace=False):
        if inplace:
            input.clamp_(min, max)
            return input
        return torch.clamp(input, min, max)

    def forward(self, x: Tensor) -> (Tensor, Tensor, Tensor):
        
        init_clip_val = self.s
        
        n = 2 ** self.bit_width -1 

        diff = init_clip_val - 0
        scale = n / diff
        zero_point = scale * 0

        scale.to(x)
        zero_point.to(x)
        bit_width = torch.Tensor([self.bit_width]).to(x)
        
        output = self.clamp(x, 0, init_clip_val.item())

        output = LearnedClippedLinearQuantizeSTE.apply(output, init_clip_val, bit_width, True, False)
        # output = self.linear_quantize(output, scale, zero_point)
        # output = self.linear_dequantize(output, scale, zero_point)
        
        return output, 1 / scale,  zero_point, bit_width
 

class PACT_weight_quant_2bits(ExtendedInjector):
    proxy_class = WeightQuantProxyFromInjector
    tensor_quant = PACT_Weight_Quantizer
    signed = True
    is_activation = False        

class PACT_activation_quant_2bits(ExtendedInjector):
    proxy_class = ActQuantProxyFromInjector
    tensor_quant = PACT_ACT_Quantizer
    signed = False
    is_activation = True

class LSQ_weight_quant_2bits(ExtendedInjector):
    proxy_class = WeightQuantProxyFromInjector
    tensor_quant = LSQ_Quantizer
    signed = True
    is_activation = False
    

# update_my_quant_injector(layer: Module, injector: Injector,  prefix: str, **kwargs) -> Injector:
#     injector = injector.let(weight_mean=layer.weight.abs().mean())  # example, pass shape of weights to injector
#     return injector

def _prep_saturation_val_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val)
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out

def linear_quantize(input, scale, zero_point, inplace=False):
    
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    if isinstance(scale, torch.Tensor):
        return torch.round(scale.to(input.device) * input - zero_point.to(input.device)) # HACK for PACT
    else:
        return torch.round(scale * input - zero_point)

def linear_dequantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    if isinstance(scale, torch.Tensor):
    # print('inside of dequantize: scale{}, zero_point{}, and input{},'.format(scale, zero_point, input)) 
        return (input + zero_point.to(input.device)) / scale.to(input.device) # HACK for PACT
    else:
        return (input + zero_point) / scale


class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input)
        output = linear_quantize(input, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None, None
    
class LearnedClippedLinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, dequantize, inplace):
        ctx.save_for_backward(input, clip_val)
        if inplace:
            ctx.mark_dirty(input)
        scale, zero_point = asymmetric_linear_quantization_params(num_bits, 0, clip_val.data, signed=False)
        
        if isinstance(clip_val, torch.Tensor):
            if input.min() < 0:
                import pdb; pdb.set_trace()
                raise ValueError('[JC] SENQNN: input to ClippedLinearQuantization should be non-negative.')
            output = torch.where(input>clip_val, torch.ones_like(input)*clip_val, input) ##naigang: to combine last two lines for speedup
        else:
            output = clamp(input, 0, clip_val.data, inplace)
        output = linear_quantize(output, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        #Naigang: modify last two lines for speedup
        grad_input = torch.where(input<0, torch.zeros_like(grad_input), grad_input) 
        grad_input = torch.where(input>clip_val, torch.zeros_like(grad_input), grad_input) 

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(input<clip_val, torch.zeros_like(grad_alpha), grad_alpha) 

        # if PRINT_TENSOR:
        #     print("grad_alpha before sum {}".format(grad_alpha))
        grad_alpha = grad_alpha.sum().expand_as(clip_val)
        
        return grad_input, grad_alpha, None, None, None


def asymmetric_linear_quantization_params(num_bits, saturation_min, saturation_max,
                                          integral_zero_point=True, signed=False):
    with torch.no_grad():                                      
        scalar_min, sat_min = _prep_saturation_val_tensor(saturation_min)
        scalar_max, sat_max = _prep_saturation_val_tensor(saturation_max)
        is_scalar = scalar_min and scalar_max

        if scalar_max and not scalar_min:
            sat_max = sat_max.to(sat_min.device)
        elif scalar_min and not scalar_max:
            sat_min = sat_min.to(sat_max.device)
       
#        print('device {}, sat_min {}'.format(sat_min.device.index, sat_min))
#        print('device {}, sat_max {}'.format(sat_min.device.index, sat_max))
        
       # if any(sat_min > sat_max):
       #     raise ValueError('saturation_min must be smaller than saturation_max, sat_min={}, sat_max={}'.format(sat_min, sat_max))

        n = 2 ** num_bits - 1

        # Make sure 0 is in the range
        sat_min = torch.min(sat_min, torch.zeros_like(sat_min))
        sat_max = torch.max(sat_max, torch.zeros_like(sat_max))

        diff = sat_max - sat_min
       # print('diff is :', diff)
        # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
        # value to 'n', so the scale becomes 1
        diff[diff == 0] = n

        scale = n / diff
        zero_point = scale * sat_min
        if integral_zero_point:
            zero_point = zero_point.round()
        if signed:
            zero_point += 2 ** (num_bits - 1)
        if is_scalar:
            return scale.item(), zero_point.item()
#        print('device {}, scale {}'.format(scale.device.index, scale))
#        print('device {}, zero_point {}'.format(zero_point.device.index, zero_point))
        return scale, zero_point