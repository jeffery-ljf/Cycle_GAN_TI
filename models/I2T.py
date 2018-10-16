# -*- coding:utf-8 -*-
import torch.nn as nn
import torch
class Encoder_Image(nn.Module):  # 把图片转换成图片特征
    def __init__(self, num_channels, bias=True):
        super().__init__()
        # 假设输入size为64*64，输出为out_channels的向量
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
    def forward(self, input):
        result_conv = self.conv_layers(input)
        return torch.squeeze(torch.squeeze(result_conv, 2), 2)  # 去掉height和width维
class Generator_I2T(nn.Module):  # 接收图片特征以生成文字的生成器
    def __init__(self, vocab_size, input_size, hidden_size, bias=True):
        super().__init__()
        # 构建GRU生成文字
        self.grucell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size, bias=bias)
        self.output_layer = nn.Sequential(*[nn.Linear(in_features=hidden_size, out_features=vocab_size, bias=bias),
                                            nn.Softmax(dim=1)])
    def forward(self, input, h_last, is_stochastic=True):
        try:
            h = self.grucell(input, h_last)  # return batch, hidden_size
            output = self.output_layer(h)  # return batch, vocab_size
            if is_stochastic:  # 采用随机概率抽样的方式
                sample = torch.squeeze(torch.multinomial(output, 1), 1)
            else:
                sample = torch.max(output, 1)[1]
            return sample, h, output  # return batch  -   batch, hidden_size   -   batch, vocab_size
        except Exception as e:
            print(repr(e))