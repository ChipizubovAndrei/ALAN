import csv

from torch import nn
import torchvision

class LoggerCSV:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, log):
        #log = [epoch, train loss, psnr, ssim]
        try:
            if log[0] == 0:
                mode = 'w'
                with open(f'./logs/{self.model_name}.csv', mode, newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Epoch", "Train loss", "Val PSNR", "Val SSIM"])
            else:
                mode = 'a'
            
            with open(f'./logs/{self.model_name}.csv', mode, newline='') as file:
                writer = csv.writer(file)
                writer.writerow(log)
            file.close()
        except FileNotFoundError:
            print("Wrong path to the log file.")

class TruncatedVGG19(nn.Module):

    def __init__(self, i, j):
        super(TruncatedVGG19, self).__init__()

        vgg19 = torchvision.models.vgg19(pretrained=True)

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

    def forward(self, input):
        output = self.truncated_vgg19(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)

        return output
