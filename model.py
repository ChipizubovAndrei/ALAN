import sys
import math

sys.path.append("custom_layers")

import torch
from torch import nn
import torch.nn.init as init

class ACBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False,
                 reduce_gamma=False, gamma_init=None ):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

            if padding - (kernel_size + (kernel_size - 1)*(dilation - 1)) // 2 >= 0:
                #   Common use case. E.g., k=3, p=1 or k=5, p=2
                self.crop = 0
                #   Compared to the KxK layer, the padding of the 1xK layer and Kx1 layer should be adjust to align the sliding windows (Fig 2 in the paper)
                hor_padding = [padding - (kernel_size + (kernel_size - 1)*(dilation - 1)) // 2, padding]
                ver_padding = [padding, padding - (kernel_size + (kernel_size - 1)*(dilation - 1)) // 2]
            else:
                #   A negative "padding" (padding - kernel_size//2 < 0, which is not a common use case) is cropping.
                #   Since nn.Conv2d does not support negative padding, we implement it manually
                self.crop = (kernel_size + (kernel_size - 1)*(dilation - 1)) // 2 - padding
                hor_padding = [0, padding]
                ver_padding = [padding, 0]

            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                      stride=stride,
                                      padding=ver_padding, dilation=dilation, groups=groups, bias=True,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                      stride=stride,
                                      padding=hor_padding, dilation=dilation, groups=groups, bias=True,
                                      padding_mode=padding_mode)

            if reduce_gamma:
                self.init_gamma(1.0 / 3)

            if gamma_init is not None:
                assert not reduce_gamma
                self.init_gamma(gamma_init)

    def _fuse_bn_tensor(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

    def get_equivalent_kernel_bias(self):
        hor_k = self.hor_conv.weight * 1
        ver_k = self.ver_conv.weight * 1
        square_k = self.square_conv.weight * 1
        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        return square_k, self.hor_conv.bias + self.ver_conv.bias + self.square_conv.bias

    def switch_to_deploy(self):
        # deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()

        self.deploy = True
        self.fused_conv = nn.Conv2d(in_channels=self.square_conv.in_channels, out_channels=self.square_conv.out_channels,
                                    kernel_size=self.square_conv.kernel_size, stride=self.square_conv.stride,
                                    padding=self.square_conv.padding, dilation=self.square_conv.dilation, groups=self.square_conv.groups, bias=True,
                                    padding_mode=self.square_conv.padding_mode)
        self.__delattr__('square_conv')
        self.__delattr__('hor_conv')
        self.__delattr__('ver_conv')
        self.fused_conv.weight.data = deploy_k
        self.fused_conv.bias.data = deploy_b

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            if self.crop > 0:
                ver_input = input[:, :, :, self.crop:-self.crop]
                hor_input = input[:, :, self.crop:-self.crop, :]
            else:
                ver_input = input
                hor_input = input
            vertical_outputs = self.ver_conv(ver_input)
            horizontal_outputs = self.hor_conv(hor_input)
            result = square_outputs + vertical_outputs + horizontal_outputs
            return result


class ACBBatchNormBlock(nn.Module):
    """
    Во время обучения ACB использует три Conv с асимметричными размерами ядра 3×3, 1×3 и 3×1 соответственно. 
    ACB переключается на стандартную структуру Conv для вывода путем объединения трех ядер и смещений Conv.
    URL: https://github.com/DingXiaoH/ACNet/blob/master/acnet/acb.py
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False,
                 use_affine=True, reduce_gamma=False, gamma_init=None ):
        super(ACBBatchNormBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)


            if padding - kernel_size // 2 >= 0:
                #   Common use case. E.g., k=3, p=1 or k=5, p=2
                self.crop = 0
                #   Compared to the KxK layer, the padding of the 1xK layer and Kx1 layer should be adjust to align the sliding windows (Fig 2 in the paper)
                hor_padding = [padding - kernel_size // 2, padding]
                ver_padding = [padding, padding - kernel_size // 2]
            else:
                #   A negative "padding" (padding - kernel_size//2 < 0, which is not a common use case) is cropping.
                #   Since nn.Conv2d does not support negative padding, we implement it manually
                self.crop = kernel_size // 2 - padding
                hor_padding = [0, padding]
                ver_padding = [padding, 0]

            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                      stride=stride,
                                      padding=ver_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                      stride=stride,
                                      padding=hor_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

            if reduce_gamma:
                self.init_gamma(1.0 / 3)

            if gamma_init is not None:
                assert not reduce_gamma
                self.init_gamma(gamma_init)


    def _fuse_bn_tensor(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

    def get_equivalent_kernel_bias(self):
        hor_k, hor_b = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
        ver_k, ver_b = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
        square_k, square_b = self._fuse_bn_tensor(self.square_conv, self.square_bn)
        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        return square_k, hor_b + ver_b + square_b


    def switch_to_deploy(self):
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.deploy = True
        self.fused_conv = nn.Conv2d(in_channels=self.square_conv.in_channels, out_channels=self.square_conv.out_channels,
                                    kernel_size=self.square_conv.kernel_size, stride=self.square_conv.stride,
                                    padding=self.square_conv.padding, dilation=self.square_conv.dilation, groups=self.square_conv.groups, bias=True,
                                    padding_mode=self.square_conv.padding_mode)
        self.__delattr__('square_conv')
        self.__delattr__('square_bn')
        self.__delattr__('hor_conv')
        self.__delattr__('hor_bn')
        self.__delattr__('ver_conv')
        self.__delattr__('ver_bn')
        self.fused_conv.weight.data = deploy_k
        self.fused_conv.bias.data = deploy_b


    def init_gamma(self, gamma_value):
        init.constant_(self.square_bn.weight, gamma_value)
        init.constant_(self.ver_bn.weight, gamma_value)
        init.constant_(self.hor_bn.weight, gamma_value)
        print('init gamma of square, ver and hor as ', gamma_value)

    def single_init(self):
        init.constant_(self.square_bn.weight, 1.0)
        init.constant_(self.ver_bn.weight, 0.0)
        init.constant_(self.hor_bn.weight, 0.0)
        print('init gamma of square as 1, ver and hor as 0')

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            if self.crop > 0:
                ver_input = input[:, :, :, self.crop:-self.crop]
                hor_input = input[:, :, self.crop:-self.crop, :]
            else:
                ver_input = input
                hor_input = input
            vertical_outputs = self.ver_conv(ver_input)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv(hor_input)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            result = square_outputs + vertical_outputs + horizontal_outputs
            return result

class ADWCA( nn.Module ):
    def __init__(self, in_channels, out_channels, use_batch_norm=False):
        super(ADWCA, self).__init__()

        if use_batch_norm:
            ACB = ACBBatchNormBlock
        else:
            ACB = ACBlock

        self.pwconv2d_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                    kernel_size=1, padding=0, dilation=1, groups=1, bias=True)
        self.gelu = nn.GELU()

        self.dwacb = ACB(in_channels=in_channels, out_channels=in_channels, kernel_size=5,
                                stride=1, padding=2, dilation=1, groups=in_channels)

        self.dwdacb = ACB(in_channels=in_channels, out_channels=in_channels, kernel_size=7,
                                stride=1, padding=9, dilation=3, groups=in_channels)

        self.pwconv2d_2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                            kernel_size=1, padding=0, dilation=1, groups=1, bias=True)

        self.pwconv2d_3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                    kernel_size=1, padding=0, dilation=1, groups=1, bias=True)

    
    def switch_to_deploy(self):
        self.dwacb.switch_to_deploy()
        self.dwdacb.switch_to_deploy()

    def forward(self, input):

        pwconv_input = self.gelu( self.pwconv2d_1( input ) )

        result = self.dwacb( pwconv_input )
        result = self.dwdacb( result )
        result = self.pwconv2d_2( result )

        result = pwconv_input * result

        result = self.pwconv2d_3( result )

        return result

class ALKCB( nn.Module ):
    def __init__(self, in_channels, use_affine=True, use_batch_norm=False):
        super( ALKCB, self ).__init__()

        if use_batch_norm:
            ACB = ACBBatchNormBlock
        else:
            ACB = ACBlock
        
        self.bn1 = nn.BatchNorm2d( num_features=in_channels, affine=use_affine )
        self.adwca1 = ADWCA( in_channels=in_channels, out_channels=in_channels, use_batch_norm=use_batch_norm  )

        self.bn2 = nn.BatchNorm2d( num_features=in_channels, affine=use_affine )

        out_channels = 4 * in_channels
        self.conv1 = nn.Conv2d( in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=1, padding=0, groups=1, dilation=1, bias=True)
        self.dwacb = ACBlock( in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                padding=1, dilation=1, groups=out_channels )

        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d( in_channels=out_channels, out_channels=in_channels, 
                                kernel_size=1, padding=0, groups=1, dilation=1, bias=True)

    def forward( self, input ):
        x = self.adwca1( self.bn1( input ) )
        output = torch.add(input, x)

        x = self.bn2( output )
        x = self.conv1( x )
        x = self.dwacb( x )
        x = self.act1( x )
        x = self.conv2( x )

        output = torch.add(output, x)
        return output

    def switch_to_deploy( self ):
        self.adwca1.switch_to_deploy()
        self.dwacb.switch_to_deploy()

class Stage( nn.Module ):
    def __init__(self, in_channels, out_channels, use_batch_norm=False):
        super( Stage, self ).__init__()

        self.alkcb1 = ALKCB( in_channels=in_channels, use_batch_norm=use_batch_norm )
        self.alkcb2 = ALKCB( in_channels=in_channels, use_batch_norm=use_batch_norm  )
        self.alkcb3 = ALKCB( in_channels=in_channels, use_batch_norm=use_batch_norm  )
        self.alkcb4 = ALKCB( in_channels=in_channels, use_batch_norm=use_batch_norm  )
        self.alkcb5 = ALKCB( in_channels=in_channels, use_batch_norm=use_batch_norm  )
        self.alkcb6 = ALKCB( in_channels=in_channels, use_batch_norm=use_batch_norm  )

    def forward(self, input):
        output = self.alkcb1(input)
        output = self.alkcb2(output)
        output = self.alkcb3(output)
        output = self.alkcb4(output)
        output = self.alkcb5(output)
        output = self.alkcb6(output)

        output = input + output
        return output

    def switch_to_deploy(self):
        self.alkcb1.switch_to_deploy()
        self.alkcb2.switch_to_deploy()
        self.alkcb3.switch_to_deploy()
        self.alkcb4.switch_to_deploy()
        self.alkcb5.switch_to_deploy()
        self.alkcb6.switch_to_deploy()

class SubPixelConv(nn.Module):
    def __init__(self, kernel_size, n_channels, scale):
        super(SubPixelConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scale ** 2),
                                kernel_size=kernel_size, padding=kernel_size // 2)
        self.shuffle = nn.PixelShuffle(upscale_factor=scale)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        x = self.act(x)
        return x

class ALAN(nn.Module):
    def __init__( self, n_channels, config, model_name='pytorch_model' ):
        self.model_name = model_name
        ACB = ACBlock
        super(ALAN, self).__init__()
        # Feature extraction module
        self.config = config
        use_batch_norm = self.config.use_batch_norm
        if use_batch_norm:
            ACB = ACBBatchNormBlock
        else:
            ACB = ACBlock
        self.acb_a1 = ACB( in_channels=3, out_channels=n_channels, kernel_size=7, padding=3 )
        self.stage_a1 = Stage( in_channels=n_channels, out_channels=n_channels, use_batch_norm=use_batch_norm )
        self.acb_a2 = ACB( in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1 )

        self.stage_a2 = nn.Sequential(
            *[Stage(in_channels=n_channels, out_channels=n_channels, use_batch_norm=use_batch_norm) for i in range(self.config.n_stage)])
        
        self.acb_a3 = ACB( in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1 )
        self.stage_a3 = Stage( in_channels=n_channels, out_channels=n_channels, use_batch_norm=use_batch_norm  )

        # Image Reconstraction Module
        self.nn_upsampling1 = nn.UpsamplingNearest2d(scale_factor=self.config.scale)

        self.acb_a4 = ACB( in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1 )
        # Upsampling

        upsampling_type = self.config.upsampling_type
        if upsampling_type == 'nearest':
            self.nn_upsampling_b1 = nn.UpsamplingNearest2d(scale_factor=self.config.scale)
        elif upsampling_type == 'subpixel':
            n_subpixel = int(math.log2(self.config.scale))
            self.nn_upsampling_b1 = nn.Sequential(
                    *[SubPixelConv(kernel_size=3, n_channels=n_channels, scale=2) for i in range(n_subpixel)])
            
        # Можно добывить еще M слоев
        self.acb_b1 = nn.Sequential(
            *[ACB( in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1 ) for i in range(self.config.n_up_acb)])

        # Pixel Attention
        self.conv_c1 = nn.Conv2d( in_channels=n_channels, out_channels=n_channels, kernel_size=1 )
        self.act_c1 = nn.Sigmoid()

        self.acb_d1 = ACB( in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1 )
        self.acb_d2 = ACB( in_channels=n_channels, out_channels=3, kernel_size=3, padding=1 )

    def forward(self, x):
        nn_output = self.nn_upsampling1(x)

        # Feature extraction module
        x = self.acb_a1(x)
        skip_connection = x

        x = self.stage_a1(x)
        x = self.acb_a2(x)
        x = self.stage_a2(x)

        x = self.acb_a3(x)
        x = self.stage_a3(x)

        x = torch.add(x, skip_connection)

        # Image reconstraction module
        x = self.acb_a4(x)
        #Upsampling
        x = self.nn_upsampling_b1(x)
        x = self.acb_b1(x)
        # Pixel attention
        skip_connection = x
        x = self.act_c1( self.conv_c1(x) )
        x = skip_connection * x

        x = self.acb_d1(x)
        x = self.acb_d2(x)

        x = torch.add(x, nn_output)

        return x

    def switch_to_deploy(self):
        self.acb_a1.switch_to_deploy()
        self.acb_a2.switch_to_deploy()
        self.acb_a3.switch_to_deploy()
        self.acb_a4.switch_to_deploy()
        for layer in self.acb_b1:
            layer.switch_to_deploy()
        self.acb_d1.switch_to_deploy()
        self.acb_d2.switch_to_deploy()

        self.stage_a1.switch_to_deploy()
        for layer in self.stage_a2:
            layer.switch_to_deploy()
        self.stage_a3.switch_to_deploy()
