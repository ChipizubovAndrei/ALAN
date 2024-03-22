from torch import nn
import torch

from acb import ACBlock, ACBBatchNormBlock
from adwca import ADWCA


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