import csv
import os.path

import torch
from torch import nn
import torchvision
import torchvision.transforms.functional as FT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

def convert_image(img, source, target):
    """
    Convert an image from a source format to a target format.

    :param img: image
    :param source: source format, one of 'pil' (PIL image), '[0, 1]' or '[-1, 1]' (pixel value ranges)
    :param target: target format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                   'imagenet-norm' (pixel values standardized by imagenet mean and std.),
                   'y-channel' (luminance channel Y in the YCbCr color format, used to calculate PSNR and SSIM)
    :return: converted image
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm',
                      'y-channel'}, "Cannot convert to target format %s!" % target

    # Convert from source to [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)

    elif source == '[0, 1]':
        pass  # already in [0, 1]

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # Convert from [0, 1] to target
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # already in [0, 1]

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # Based on definitions at https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.

    return img

class LoggerCSV:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, log):
        #log = [epoch, val loss, psnr, ssim]
        try:
            if not os.path.exists(f'./logs/{self.model_name}.csv'):
                mode = 'w+'
                file = open(f'./logs/{self.model_name}.csv', mode, newline='')
                writer = csv.writer(file)
                writer.writerow(["Epoch", "Train loss", "Val PSNR", "Val SSIM"])
            else:
                mode = 'a'
            
            with open(f'./logs/{self.model_name}.csv', mode, newline='') as file:
                writer = csv.writer(file)
                writer.writerow(log)
        except FileNotFoundError:
            print("Wrong path to the log file.")

class TruncatedVGG19(nn.Module):

    def __init__(self, i, j):
        super(TruncatedVGG19, self).__init__()

        vgg19 = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        for layer in vgg19.features.children():
            truncate_at += 1

            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            if maxpool_counter == i - 1 and conv_counter == j:
                break

        assert maxpool_counter == i - 1 and conv_counter == j, "One or both of i=%d and j=%d are not valid choices for the VGG19!" % (
            i, j)

        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])
        for param in self.truncated_vgg19.parameters(): param.requires_grad = False

    def forward(self, input):
        output = self.truncated_vgg19(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)

        return output
