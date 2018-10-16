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
                nn.Tanh()
            ]
            self.convs.append(nn.Sequential(*model))
        total_features = sum(num_filters) + 1720  # 加入噪声
        # 把得到的文字特征，反卷积成一张图片
        self.convTranspose_layers = nn.Sequential(*[nn.ConvTranspose2d(in_channels=total_features, out_channels=256,
                                                                       kernel_size=(16, 16), stride=16),
                                                    nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                                                       kernel_size=(4, 4), stride=4),
                                                    nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                                                       kernel_size=(4, 4), stride=1),
                                                    nn.Conv2d(in_channels=256, out_channels=3,
                                                              kernel_size=(4, 4), stride=1),
                                                    nn.Sigmoid()
                                                    ])

    def forward(self, input):
        result = []
        for i in range(self.convs.__len__()):
            result_conv = self.convs[i](input)
            pool_layer = nn.MaxPool2d(kernel_size=(input.shape[2] - self.filter_sizes[i] + 1, 1))
            result.append(pool_layer(result_conv))
        h_pooled_flat = torch.cat(result, dim=1)
        #加入抽样的向量
        samples = []
        for i in range(input.shape[0]):
            sample = torch.ones([1, 1720, 1, 1]).normal_(mean=0, std=10)
            samples.append(sample)
        samples = torch.cat(samples, dim=0).cuda()
        samples = torch.cat([h_pooled_flat, samples], dim=1)
        print(torch.norm((samples[0]-samples[1]), p=2))
        #经过反卷积层
        output = self.convTranspose_layers(samples)
        return output  # return batch, 3, height, width
class Discriminator_T2I(nn.Module):
    def __init__(self, num_channels, bias=True):
        super().__init__()
        # 构建卷积层获取图像的信息
        self.conv_layers = nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels=num_channels, kernel_size=(5, 5),
                                                     stride=1, bias=bias),
                                           nn.ReLU(True),
                                           nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(4, 4),
                                                     stride=4, bias=bias),
                                           nn.ReLU(True),
                                           nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(2, 2),
                                                     stride=1, bias=bias),
                                           nn.BatchNorm2d(num_features=num_channels),
                                           nn.ReLU(True),
                                           nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(2, 2),
                                                     stride=2, bias=bias),
                                           nn.ReLU(True),
                                           nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(2, 2),
                                                     stride=1, bias=bias),
                                           nn.BatchNorm2d(num_features=num_channels),
                                           nn.ReLU(True),
                                           nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(2, 2),
                                                     stride=2, bias=bias),
                                           nn.ReLU(True),
                                           nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3),
                                                     stride=1, bias=bias),
                                           nn.ReLU(True)
                                           ])
        # Highway Layer
        self.highway_linear_H = nn.Sequential(
            *[nn.Linear(in_features=num_channels, out_features=num_channels, bias=True),
              nn.ReLU(True)])
        self.highway_linear_G = nn.Sequential(
            *[nn.Linear(in_features=num_channels, out_features=num_channels, bias=True),
              nn.Sigmoid()])
        # 输出层
        self.output_layer = nn.Sequential(*[nn.Linear(in_features=num_channels, out_features=1, bias=bias),
                                            nn.Sigmoid()])
    def __highway(self, input_, num_layers=1):
        """Highway Network (cf. http://arxiv.org/abs/1505.00387).
            t = sigmoid(Wy + b)
            z = t * g(Wy + b) + (1 - t) * y
            where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
        """
        for idx in range(num_layers):
            g = self.highway_linear_H(input_)
            t = self.highway_linear_G(input_)
            output = t*g+(1.-t)*input_
            return output
    def forward(self, input):
        result_conv = self.conv_layers(input)
        result_conv = torch.squeeze(result_conv, 2)
        result_conv = torch.squeeze(result_conv, 2)
        h_highway = self.__highway(input_=result_conv, num_layers=1)
        result_output = self.output_layer(h_highway)
        return result_output