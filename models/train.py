# -*- coding:utf-8 -*-
import time
import torch
import torch.nn as nn
import visdom
from models.cyclegan_TI import CycleGAN_TI
from torch.nn import init
def validate_G(texts, epoch, win, vis):
    text_list = model.dataset.show(texts)
    text_vis = win + str(epoch) + ': '
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

    model.load_state_dict(torch.load('../save/fix_GT2I&emb_2510.pkl'))
    is_updateD = True
    for epoch in range(10000):
        time1 = time.time()
        totalloss_rsc_T2I = 0.0
        totalloss_Gads_T2I = 0.0
        totalloss_D_T2I = 0.0
        total_J_I2T = 0.0
        totalloss_D_I2T = 0.0
        total_J_rsc_I2T = 0.0
        totalloss_rsc_rewards_I2T = 0.0
        for i, batch in enumerate(model.dataloader):
            texts = batch[0].cuda()
            images = batch[1].cuda()
            seqlen = batch[2].cuda()  # 记录每个句子的实际长度
            loss_rsc_T2I, loss_Gads_T2I, loss_D_T2I, texts_rsc, images_gen, J_I2T, loss_D_I2T, J_rsc_I2T, loss_rsc_rewards_I2T, texts_gen = model.forward((texts, images, seqlen))
            if is_updateD:
                model.backward(updateD=True)
            else:
                model.backward(updateD=False)
            totalloss_rsc_T2I += loss_rsc_T2I
            totalloss_Gads_T2I += loss_Gads_T2I
            totalloss_D_T2I += loss_D_T2I
            total_J_I2T += J_I2T
            totalloss_D_I2T +=loss_D_I2T
            total_J_rsc_I2T += J_rsc_I2T
            totalloss_rsc_rewards_I2T += loss_rsc_rewards_I2T
            for j in range(images_gen.shape[0]):
                vis.image(img=images_gen[j], win='image' + str(j))
            validate_G(texts, epoch, 'T', vis)
            validate_G(texts_rsc, epoch, 'T_rsc', vis)
            validate_G(texts_gen, epoch, 'T_GEN', vis)
        totalloss_rsc_T2I /= (i + 1)
        totalloss_Gads_T2I /= (i + 1)
        totalloss_D_T2I /= (i + 1)
        total_J_I2T /= (i + 1)
        totalloss_D_I2T /= (i + 1)
        total_J_rsc_I2T /= (i + 1)
        totalloss_rsc_rewards_I2T /= (i + 1)
        if totalloss_D_T2I > 0.0:
            is_updateD = True
        else:
            is_updateD = False
        vis.line(X=torch.cat([torch.Tensor([[epoch]]), torch.Tensor([[epoch]]), torch.Tensor([[epoch]]),
                              torch.Tensor([[epoch]]), torch.Tensor([[epoch]]), torch.Tensor([[epoch]]),
                              torch.Tensor([[epoch]])], 1),
                 Y=torch.cat([torch.Tensor([[totalloss_rsc_T2I]]), torch.Tensor([[totalloss_Gads_T2I]]),
                              torch.Tensor([[totalloss_D_T2I]]), torch.Tensor([[total_J_I2T]]),
                              torch.Tensor([[totalloss_D_I2T]]), torch.Tensor([[total_J_rsc_I2T]]),
                              torch.Tensor([[totalloss_rsc_rewards_I2T]])], 1),
                 win='loss',
                 opts=dict(legend=['loss_rsc_T2I', 'loss_Gads_T2I', 'loss_D_T2I', 'J_I2T', 'loss_D_I2T', 'J_rsc_I2T', 'loss_rsc_rewards_I2T']),
                 update='append' if epoch > 0 else None)
        print('epoch ' + str(epoch) + ', time: ' + str(time.time()-time1))
        print('loss_rsc_T2I: ' + str(totalloss_rsc_T2I))
        print('loss_Gads_T2I: ' + str(totalloss_Gads_T2I))
        print('loss_D_T2I: ' + str(totalloss_D_T2I))
        if epoch % 10 ==0:
            torch.save(model.state_dict(), '../save/complete_' + str(epoch) + '.pkl')
    print()