import torch
import torch.nn as nn

architecture_config = [(7, 64, 2, 3), "M"]


class CNNBlock(nn.Module):
    def __int__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__int__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakrelu = nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.leakrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels,  **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlock(
                    in_channels, out_channels = x[1], kernel_size = x[0], stride = x[2], padding = x[3]
                )
                ]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv1 = x[0] # tuple
                conv2 = x[1] # tuple
                num_repeats = x[2]
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels, conv1[1], conv1[0], conv1[2], conv1[3]
                        )
                    ]
                    layers += [
                        CNNBlock(
                            in_channels, conv2[1], conv2[0], conv2[2], conv2[3]
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S, 4960),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4960, S*S*(C+B*C))
        )





def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
