# -*- coding:utf-8 -*-
import torch
import visdom
from torch.nn import init
from torch.autograd import Variable

from models.cyclegan_TI import CycleGAN_TI

# model = CycleGAN_TI().cuda()
# # model.load_state_dict(torch.load('../save/completed_30.pkl'))
# vis = visdom.Visdom(port=2424, env='test')
# while True:
#     for i, batch in enumerate(model.dataloader):
#         texts = batch[0].cuda()
#         images = batch[1].cuda()
#         for j in range(images.shape[0]):
#             vis.image(img=images[j][0], win='image' + str(j))
#         y_gen = model.generate_Y(texts)
v1 = Variable(torch.tensor([1.0, 2.0, 3.0]))
t1 = torch.tensor([1.0, 2.0, 3.0])
t1.requires_grad = True
t2 = torch.tensor([1.0, 2.0, 3.0])
t3 = v1 + t2
sum(t3).backward()
print()
