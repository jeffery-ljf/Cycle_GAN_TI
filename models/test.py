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
#     for i, batch in enumerate(model.dataloader_pretraining):
#         images = batch[0].cuda()
#         texts = batch[1].cuda()
#         for j in range(1):
#             vis.image(img=images[j][0], win='image' + str(j))
#             print(texts[j])
# dict = {}
# dict['a'] = {'a1': 1, 'a2': 2, 'a3': 3}
# dict['b'] = {'b1': 1, 'b2': 2, 'b3': 3}
# dict['c'] = {'c1': 1, 'c2': 2, 'c3': 3}
x = torch.Tensor([2, 3, 4])
print("""
hahaha
xixixi
hehehe
""")
