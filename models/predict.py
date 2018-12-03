# -*- coding:utf-8 -*-
import torch
import visdom

from models.cyclegan_TI import CycleGAN_TI
from models.train import init_parameters

if __name__ == '__main__':
    vis = visdom.Visdom(port=2424, env='predict')
    model = CycleGAN_TI().cuda()
    init_parameters(model)  # 初始化参数
    model.load_state_dict(torch.load('../save/fix_GT2I&emb_2500.pkl'))
    for i, batch in enumerate(model.dataloader):
        texts = batch[0].cuda()[:1]
        images = batch[1].cuda()[:1]
        seqlen = batch[2].cuda()[:1]  # 记录每个句子的实际长度
        # 测试T2I
        # y_gen = model.generate_Y(texts, seqlen)
        # vis.image(img=y_gen[0], win='image')

        # 测试I2T
        samples, hs, predictions, record, now_len = model.generate_X_nofixedlen(images, model.start_input[:1],
                                                                                model.start_h[:1], 20)  # 输入图片，生成句子
        trans_samples = torch.transpose(samples, dim0=0, dim1=1)  # batch * seq_len
        vis.image(img=images[0], win='image')
        print()

