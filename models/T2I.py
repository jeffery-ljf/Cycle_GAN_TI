# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
class Generator_T2I(nn.Module):  # 接收文字以生成图片的生成器
    def __init__(self, embedding_size, filter_sizes, num_filters, bias=True):
        super().__init__()
        self.num_trans_features = 128
        '''
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
        num_text_features = sum(num_filters)
        '''
        self.GRU = nn.GRU(input_size=embedding_size, hidden_size=embedding_size, num_layers=1, batch_first=True, bias=bias)
        self.transform_layer = nn.Sequential(*[nn.Linear(in_features=embedding_size, out_features=self.num_trans_features, bias=bias),  # 转换文本的特征
                                               nn.Tanh()])
        num_total_features = self.num_trans_features + 128  # 加入噪声
        # 把得到的文字特征，反卷积成一张图片
        self.convTranspose_layers = nn.Sequential(*[nn.ConvTranspose2d(in_channels=num_total_features, out_channels=512,
                                                                       kernel_size=(7, 7), stride=7, bias=bias),
                                                    nn.ConvTranspose2d(in_channels=512, out_channels=1,
                                                                       kernel_size=(4, 4), stride=4),
                                                    # nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(3, 3),
                                                    #           stride=1, bias=bias),
                                                    # nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(1, 1),
                                                    #           stride=1, bias=bias),

                                                    # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2, 2),
                                                    #           stride=1),
                                                    # nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=(3, 3),
                                                    #           stride=1, padding=1),
                                                    nn.Sigmoid()
                                                    ])

    def forward(self, input, seqlen):
        '''

        :param input: 输入的单词序列向量: batch * seq_len * embedding_size
        :param seqlen: 每条单词序列的实际长度:batch
        :return:
        '''
        result = []
        '''
        for i in range(self.convs.__len__()):
            result_conv = self.convs[i](input)
            pool_layer = nn.MaxPool2d(kernel_size=(input.shape[2] - self.filter_sizes[i] + 1, 1))
            result.append(pool_layer(result_conv))
        h_pooled_flat = torch.squeeze(torch.squeeze(torch.cat(result, dim=1), dim=2), dim=2)  # batch * num_text_features
        '''
        for i in range(input.shape[0]):
            _, h_n = self.GRU(input[i:i+1, :seqlen[i]])
            result.append(torch.squeeze(h_n, dim=0))
        h_pooled_flat = torch.cat(result, dim=0)
        trans_features = torch.unsqueeze(torch.unsqueeze(self.transform_layer(h_pooled_flat), dim=2), dim=2)  # batch * num_trans_features * 1 * 1
        # trans_features = self.transform_layer(h_pooled_flat) # batch * num_trans_features
        #加入抽样的向量
        samples = []
        for i in range(input.shape[0]):
            sample = torch.ones([1, 128, 1, 1]).normal_(mean=0, std=1)
            # samples.append(sample.repeat(7, 7, 1, 1).permute(2, 3, 0, 1))
            samples.append(sample)
        samples = torch.cat(samples, dim=0).cuda()

        #经过反卷积层
        # trans_features_space = trans_features.repeat(7, 7, 1, 1)  # Spatial Tiling
        # trans_features_space = trans_features_space.permute(2, 3, 0, 1)  # batch * channel * H * W
        samples = torch.cat([trans_features, samples], dim=1)
        output = self.convTranspose_layers(samples)
        return output  # return batch, 3, height, width
class Discriminator_T2I(nn.Module):
    def __init__(self, num_channels, bias=True):
        super().__init__()
        # 构建卷积层获取图像的信息
        self.conv_layers = nn.Sequential(*[nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=(5, 5),
                                                     stride=1, bias=bias),
                                           nn.ReLU(True),
                                           nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(4, 4),
                                                     stride=4, bias=bias),
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