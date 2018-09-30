# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
class Generator_T2I(nn.Module):  # 接收文字以生成图片的生成器
    def __init__(self, embedding_size, filter_sizes, num_filters):
        super().__init__()
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.convs = nn.ModuleList()
        for filter_size, num_filter in zip(filter_sizes, num_filters):
            # Convolution Layer and Pool Layer
            model = [
                nn.Conv2d(in_channels=1, out_channels=num_filter, kernel_size=(filter_size, embedding_size), stride=1),
                nn.ReLU(True)
            ]
            self.convs.append(nn.Sequential(*model))
        total_features = sum(num_filters)
        # 把得到的文字特征，反卷积成一张图片
        self.convTranspose_layers = nn.Sequential(*[nn.ConvTranspose2d(in_channels=total_features, out_channels=256,
                                                                       kernel_size=(4, 4), stride=4),
                                                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2, 2),
                                                              stride=2),
                                                    nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                                                       kernel_size=(4, 4), stride=4),
                                                    nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                                                       kernel_size=(2, 2), stride=2),
                                                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2, 2),
                                                              stride=2),
                                                    nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                                                       kernel_size=(4, 4), stride=4),
                                                    nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                                                       kernel_size=(4, 4), stride=4),
                                                    nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                                                       kernel_size=(4, 4), stride=4),
                                                    nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(2, 2),
                                                              stride=2),
                                                    nn.Sigmoid()
                                                    ])

    def forward(self, input):
        result = []
        for i in range(self.convs.__len__()):
            result_conv = self.convs[i](input)
            pool_layer = nn.MaxPool2d(kernel_size=(input.shape[2] - self.filter_sizes[i] + 1, 1))
            result.append(pool_layer(result_conv))
        h_pooled_flat = torch.cat(result, dim=1)
        # 经过反卷积层
        output = self.convTranspose_layers(h_pooled_flat)
        return output  # return batch, 3, height, width
class Discriminator_T2I(nn.Module):
    def __init__(self, num_channels, bias=True):
        super().__init__()
        # 构建卷积层获取图像的信息
        self.conv_layers = nn.Sequential(*[nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=(4, 4),
                                                     stride=4, bias=bias),
                                           nn.ReLU(True),
                                           nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(4, 4),
                                                     stride=4, bias=bias),
                                           nn.ReLU(True),
                                           nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(4, 4),
                                                     stride=4, bias=bias),
                                           nn.ReLU(True),
                                           nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(4, 4),
                                                     stride=4, bias=bias),
                                           nn.ReLU(True)
                                           ])
        # 输出层
        self.output_layer = nn.Sequential(*[nn.Linear(in_features=num_channels, out_features=1, bias=True),
                                            nn.Sigmoid()])
    def forward(self, input):
        result_conv = self.conv_layers(input)
        result_conv = torch.squeeze(result_conv, 2)
        result_conv = torch.squeeze(result_conv, 2)
        result_output = self.output_layer(result_conv)
        return result_output