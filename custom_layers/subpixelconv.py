from torch import nn

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