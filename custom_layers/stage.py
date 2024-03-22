from torch import nn
from alkcb import ALKCB

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

