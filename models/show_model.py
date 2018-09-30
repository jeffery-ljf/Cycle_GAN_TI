# -*- coding:utf-8 -*-
import torch
import torchvision
from tensorboardX import SummaryWriter
from models.I2T import Encoder_Image, Generator_I2T
from models.T2I import Generator_T2I
from models.cyclegan_TI import CycleGAN_TI
net1 = Encoder_Image(num_channels=100).cuda()
net2 = Generator_I2T(vocab_size=1000, input_size=200, hidden_size=100).cuda()
net3 = Generator_T2I(embedding_size=100, filter_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20], num_filters=[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]).cuda()
model = CycleGAN_TI().cuda()
writer1 = SummaryWriter(log_dir='../logs1', comment='Encoder_Image')
writer2 = SummaryWriter(log_dir='../logs2', comment='Generator_T2I')

images = torch.ones([9, 3, 64, 64]).cuda()
text_input = torch.ones([9, 1, 20, 100]).cuda()
with writer1:
    writer1.add_graph(net1, input_to_model=(images, ), verbose=True)
with writer2:
    writer2.add_graph(net3, input_to_model=(text_input, ), verbose=True)
