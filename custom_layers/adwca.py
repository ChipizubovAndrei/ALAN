from torch import nn

from acb import ACBlock, ACBBatchNormBlock
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


