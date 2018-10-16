# -*- coding:utf-8 -*-
import torch
import visdom
from torch.nn import init
from torch.autograd import Variable

from models.cyclegan_TI import CycleGAN_TI

model = CycleGAN_TI().cuda()
# model.load_state_dict(torch.load('../save/completed_30.pkl'))
vis = visdom.Visdom(port=2424, env='test')
while True:
    for i, batch in enumerate(model.dataloader):
        texts = batch[0].cuda()
        images = batch[1].cuda()
        for j in range(images.shape[0]):
            vis.image(img=images[j][0], win='image' + str(j))
        y_gen = model.generate_Y(texts)
print()
