import sys
import math

sys.path.append("custom_layers")

import torch
from torch import nn

from custom_layers.acb import ACBlock, ACBBatchNormBlock
from custom_layers.stage import Stage
from custom_layers.subpixelconv import SubPixelConv

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
                    *[SubPixelConv(kernel_size=3, n_channels=n_channels, scale_factor=2) for i in range(n_subpixel)])
            
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
