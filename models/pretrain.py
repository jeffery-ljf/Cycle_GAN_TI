# -*- coding:utf-8 -*-
import time

import torch
import visdom

from models.cyclegan_TI import CycleGAN_TI
from models.train import init_parameters

if __name__ == '__main__':
    vis = visdom.Visdom(port=2424, env='cyclegan-pretrain')
    model = CycleGAN_TI().cuda()
    init_parameters(model)  # 初始化参数
    vis2 = visdom.Visdom(port=2424, env='test')
    for epoch in range(200):  # 预训练T2I生成器
        time1 = time.time()
        totalloss_pretraining_T2I = 0.0
        for i, batch in enumerate(model.dataloader_pretraining):
            images = batch[0].cuda()
            texts = batch[1].cuda()
            seqlen = batch[2].cuda()  # 记录每个句子的实际长度
            loss_pretraining_T2I, images_gen = model.pretraining(images, texts, seqlen)
            for j in range(images_gen.shape[0]):  # 显示生成的图像
                vis.image(img=images_gen[j], win='image' + str(j))
            totalloss_pretraining_T2I += loss_pretraining_T2I
            totalloss_pretraining_T2I /= (i + 1)
        vis.line(X=torch.Tensor([[epoch]]),
                 Y=torch.Tensor([[totalloss_pretraining_T2I]]),
                 win='loss_pretraining',
                 opts=dict(legend=['loss_pretraining_T2I']),
                 update='append' if epoch > 0 else None)
        print('epoch ' + str(epoch) + ', time: ' + str(time.time() - time1))
        torch.save(model.state_dict(), '../save/pretrained'+ str(epoch) +'.pkl')