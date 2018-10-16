# -*- coding:utf-8 -*-
import time
import torch
import torch.nn as nn
import visdom
from models.cyclegan_TI import CycleGAN_TI
from torch.nn import init
def validate_G(texts, epoch, win, vis):
    text_list = model.dataset.show(texts)
    text_vis = str(epoch) + ': '
    for text in text_list:
        text_vis += '<h5>'
        text_vis += ' '.join(text)
        text_vis += '</h5>'
    vis.text(text=text_vis, win=win)
def init_parameters(model):
    for weight in model.parameters():
        if weight.shape.__len__()>2:
            init.xavier_normal_(weight)
if __name__ == '__main__':
    vis = visdom.Visdom(port=2424, env='cyclegan')
    model = CycleGAN_TI().cuda()
    init_parameters(model)  # 初始化参数
    for epoch in range(10000):
        time1 = time.time()
        totalloss_rsc_T2I = 0.0
        totalloss_Gads_T2I = 0.0
        totalloss_D_T2I = 0.0
        for i, batch in enumerate(model.dataloader):
            texts = batch[0].cuda()
            images = batch[1].cuda()
            loss_rsc_T2I, loss_Gads_T2I, loss_D_T2I, texts_rsc, images_gen = model.forward((texts, images))
            if i % 2 ==0:
                model.backward(updateD=True)
            else:
                model.backward(updateD=False)
            totalloss_rsc_T2I += loss_rsc_T2I
            totalloss_Gads_T2I += loss_Gads_T2I
            totalloss_D_T2I += loss_D_T2I
            for j in range(images_gen.shape[0]):
                vis.image(img=images_gen[j], win='image' + str(j))
            validate_G(texts, epoch, 'T', vis)
            validate_G(texts_rsc, epoch, 'T_rsc', vis)
        totalloss_rsc_T2I /= (i + 1)
        totalloss_Gads_T2I /= (i + 1)
        totalloss_D_T2I /= (i + 1)
        vis.line(X=torch.cat([torch.Tensor([[epoch]]), torch.Tensor([[epoch]]), torch.Tensor([[epoch]])], 1),
                 Y=torch.cat([torch.Tensor([[totalloss_rsc_T2I]]), torch.Tensor([[totalloss_Gads_T2I]]), torch.Tensor([[totalloss_D_T2I]])], 1),
                 win='loss',
                 opts=dict(legend=['loss_rsc_T2I', 'loss_Gads_T2I', 'loss_D_T2I']),
                 update='append' if epoch > 0 else None)
        print('epoch ' + str(epoch) + ', time: ' + str(time.time()-time1))
        print('loss_rsc_T2I: ' + str(totalloss_rsc_T2I))
        print('loss_Gads_T2I: ' + str(totalloss_Gads_T2I))
        print('loss_D_T2I: ' + str(totalloss_D_T2I))
        # torch.save(model.state_dict(), '../save/completed_' + str(epoch) + '.pkl')
    print()