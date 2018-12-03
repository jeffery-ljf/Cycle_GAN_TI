# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import visdom

from models.I2T import Encoder_Image, Generator_I2T, Discriminator_I2T
from models.T2I import Generator_T2I, Discriminator_T2I
from torch.utils.data import DataLoader
from torchvision import transforms
from datas.dataset import Dataset_flickr30K, Dataset_SketchyScene, Dataset_Mnist, Dataset_Mnist_pretraining
class CycleGAN_TI(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_size = 64  # batch的大小，为1的时候，过程有使用squeeze，可能会出错
        self.num_channels_encoder = 100  # 图像encoder里的频道数
        self.num_channels_DT2I = 100  # T2I判别器中的频道数
        self.embedding_size = 100  # 单词embedding大小
        self.hidden_size_gru = 100  # gru中的隐藏层结点数
        self.height = 28  # 图像的高度
        self.width = 28  # 图像的宽度
        self.start_idx = 0  # 开始token的序号
        self.end_idx = 1  # 结束token的序号
        self.padding_idx = 2  # 填充token的序号
        self.start_input = torch.zeros(self.batch_size, self.embedding_size).cuda()  # I2T_Generator开始的输入
        self.start_h = torch.zeros(self.batch_size, self.hidden_size_gru).cuda()  # I2T_Generator开始的状态
        self.filter_sizes = [1, 2, 3, 4, 5, 6]  # 生成器T2I的窗口大小（也即每个窗口包含多少个单词）
        self.num_filters = [100, 200, 200, 200, 200, 100]  # 生成器T2I的channels数量
        self.minseqlen = 6  # 为防止TextCNN的卷积层窗口太大，应该最小的序列长度
        self.encoder_image = Encoder_Image(num_channels=self.num_channels_encoder)
        self.dataset = Dataset_Mnist(images_src='../datas/mnist/train-images.idx3-ubyte',
                                         texts_src='../datas/mnist/raw',
                                         transform=transforms.Compose([transforms.ToTensor()]),
                                         start_idx=self.start_idx, end_idx=self.end_idx, padding_idx=self.padding_idx)  # 载入数据集
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True, num_workers=4)
        self.dataset_pretraining = Dataset_Mnist_pretraining(images_src='../datas/mnist/train-images.idx3-ubyte',
                                                             labels_src='../datas/mnist/train-labels.idx1-ubyte',
                                                             texts_src='../datas/mnist/raw',
                                                             transform=transforms.Compose([transforms.ToTensor()]),
                                                             start_idx=self.start_idx, end_idx=self.end_idx,
                                                             padding_idx=self.padding_idx)
        self.dataloader_pretraining = DataLoader(dataset=self.dataset_pretraining, batch_size=self.batch_size,
                                                 shuffle=True, num_workers=4)  # 载入预训练数据集
        self.vocab_size = self.dataset.dictionary.__len__()  # 字典大小
        self.maxseqlen = self.dataset.maxseqlen  # 真实数据集的最大序列长度(算上start token和end token)
        self.G_I2T = Generator_I2T(vocab_size=self.vocab_size, input_size=self.num_channels_encoder + self.embedding_size, hidden_size=self.hidden_size_gru)
        self.G_T2I = Generator_T2I(embedding_size=self.embedding_size, filter_sizes=self.filter_sizes, num_filters=self.num_filters)
        self.D_I2T = Discriminator_I2T(embedding_size=self.embedding_size, hidden_size=self.hidden_size_gru)
        self.D_T2I = Discriminator_T2I(num_channels=self.num_channels_DT2I)
        self.embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size, padding_idx=self.padding_idx)
        # self.T2I_rsc_optimizer = torch.optim.Adam([
        #     {'params': self.G_T2I.parameters()},
        #     {'params': self.encoder_image.parameters()},
        #     {'params': self.G_I2T.parameters()},
        #     {'params': self.embeddings.parameters()}
        # ])
        # self.T2I_D_optimizer = torch.optim.Adam([
        #     {'params': self.D_T2I.parameters()}
        # ])
        # self.T2I_pretraining_optimizer = torch.optim.Adam([
        #     {'params': self.G_T2I.parameters()},
        #     {'params': self.embeddings.parameters()}
        # ])
        # self.I2T_J_optimizer = torch.optim.Adam([
        #     {'params': self.encoder_image.parameters()},
        #     {'params': self.G_I2T.parameters()},
        #     {'params': self.embeddings.parameters()}
        # ])
        # self.I2T_D_optimizer = torch.optim.Adam([
        #     {'params': self.D_I2T.parameters()}
        # ])
        self.T2I_pretraining_optimizer = torch.optim.Adam([
            {'params': self.G_T2I.parameters()},
            {'params': self.embeddings.parameters()}
        ])
        self.G_optimizer = torch.optim.Adam([
            {'params': self.G_T2I.parameters()},
            {'params': self.encoder_image.parameters()},
            {'params': self.G_I2T.parameters()},
            {'params': self.embeddings.parameters()}
        ])
        self.D_optimizer = torch.optim.Adam([
            {'params': self.D_T2I.parameters()},
            {'params': self.D_I2T.parameters()}
        ])
    def collate_fn(self, x):  # 用于dataloader对每一个batch的数据进行处理
        #处理文本
        seqlen = [line[1].__len__() for line in x]  # 记录每个句子的实际长度
        seqlen = torch.LongTensor(seqlen)
        max_seqlen = max(seqlen)  # 该batch的最大序列长度
        if max_seqlen >= self.minseqlen:  # 判断该batch的最大序列长度是否大于最大的卷积窗口
            for line in x:
                while line[1].__len__() != max_seqlen:  # 为doc_data补全1，直到所有句子的最大长度+1为止。
                    line[1].append(2)
        else:
            for line in x:
                while line[1].__len__() != self.minseqlen:  # 为doc_data补全1，直到所有句子的最大长度+1为止。
                    line[1].append(2)
        texts = [line[1] for line in x]
        texts = torch.LongTensor(texts)
        #处理图片
        images = [torch.unsqueeze(line[0], 0) for line in x]
        images = torch.cat(images, 0)
        return texts, images, seqlen  # return:batch * seq_len  -  batch * channels * height * width  -  batch
    def pad_data(self, samples, record, sequence_length):  # 对数据进行padding
        '''
        :param samples:seq_len * batch
        :param record:dictionary
        :return:
        '''
        for b in record.keys():
            for t in range(record[b]+1, sequence_length):
                samples[t][b] = 2
        return samples
    def generate_X_nofixedlen(self, image_input, start_input, start_h, maxseqlen):#生成器生成不定长的句子（会使用padding token进行填充）
        '''
        :param image_input: batch * channels * height * width
        :param start_input: batch * embedding_size
        :param start_h: batch * hidden_size
        :param maxseqlen: int
        :return:
        samples: seq_len * batch每一个时间步抽样出的单词
        hs: seq_len * batch * hidden_size每一个时间步的隐藏状态
        predictions: seq_len * batch * vocab_size每一个时间步的输出向量
        record: dict每一个batch中end_token的位置
        now_len: int最大序列长度
        '''
        record = {}#记录已经生成出end token的batch idx,以及对应在samples中end token的位置序号
        now_len = 0#记录最新生成的长度
        samples = []
        predictions = []
        hs = []
        image_features = self.encoder_image(image_input)  # batch * num_channels_encoder
        input = torch.cat([start_input, image_features], dim=1)  # 设置初始输入,batch * input_size(self.embedding_size + self.num_channels_encoder)，融入图片特征
        last_h = start_h  # 设置初始状态
        while record.__len__() != start_h.shape[0]:#判断是否所有batch都生成初end token
            # 迭代GRU
            next_token, h, prediction = self.G_I2T(input, last_h)  # 获得当前时间步预测的下一个token，隐藏状态和预测层
            samples.append(torch.unsqueeze(next_token, dim=0))
            hs.append(torch.unsqueeze(h, dim=0))
            predictions.append(torch.unsqueeze(prediction, dim=0))
            input = torch.cat([self.embeddings(next_token), image_features], dim=1)  # 融入图片特征
            last_h = h
            for i in range(next_token.shape[0]):#判断每一个next token是否end token
                if next_token[i] == self.end_idx and i not in record.keys():
                    record[i] = now_len
            now_len += 1
            if now_len ==20:  # 限制最大生成长度
                for i in range(image_input.shape[0]):
                    if i not in record.keys():
                        record[i] = now_len-1
                break
        samples = torch.cat(samples, dim=0)
        hs = torch.cat(hs, dim=0)
        predictions = torch.cat(predictions, dim=0)
        samples = self.pad_data(samples=samples, record=record, sequence_length=now_len)#对生成出来的token的end token后的位置进行padding。
        return samples, hs, predictions, record, now_len  # return seq_len, batch  -   seq_len, batch, hidden_size   -   seq_len, batch, vocab_size, dict, int
    def generate_X(self, images, start_input, start_h, sequence_length):  # 生成样本，有最大长度
        '''

        :param images: batch * num_channels_encoder * height * width
        :param start_input: batch * embedding_size
        :param start_h: batch * hidden_size
        :param sequence_length: int
        :return:samples: seq_len * batch||hs: seq_len * batch * hidden_size||predictions: seq_len * batch * vocab_size
        '''
        record = {}  # 记录已经生成出end token的batch idx,以及对应在samples中end token的位置序号
        now_len = 0  # 记录最新生成的长度
        samples = []
        predictions = []
        hs = []
        image_features = self.encoder_image(images)  # batch * num_channels_encoder
        input = torch.cat([start_input, image_features], dim=1)  # 设置初始输入,batch * input_size(self.embedding_size + self.num_channels_encoder)，融入图片特征
        last_h = start_h  # 设置初始状态
        for i in range(sequence_length):
            # 迭代GRU
            next_token, h, prediction = self.G_I2T(input, last_h)  # 获得当前时间步预测的下一个token，隐藏状态和预测层
            samples.append(torch.unsqueeze(next_token, dim=0))
            hs.append(torch.unsqueeze(h, dim=0))
            predictions.append(torch.unsqueeze(prediction, dim=0))
            input = torch.cat([self.embeddings(next_token), image_features], dim=1)  # 融入图片特征
            last_h = h
            for i in range(next_token.shape[0]):  # 判断每一个next token是否end token
                if next_token[i] == 1 and i not in record.keys():
                    record[i] = now_len
            now_len += 1
        samples = torch.cat(samples, dim=0)
        hs = torch.cat(hs, dim=0)
        predictions = torch.cat(predictions, dim=0)
        samples = self.pad_data(samples=samples, record=record, sequence_length = sequence_length)  # 对生成出来的token的end token后的位置进行padding。
        return samples, hs, predictions, record  # return seq_len * batch  -   seq_len * batch * hidden_size   -   seq_len * batch * vocab_size   -  list
    def generate_pretrained(self, image_input, start_input, start_h, sequence_length, groundtrues):#输入图片，输出文字
        '''

        :param start_input: batch * embedding_size
        :param start_h: batch * hidden_size
        :param sequence_length: int
        :param groundtrues: sequence_length * batch
        :return:predictions: seq_len * batch * vocab_size
        '''
        predictions = []
        image_features = self.encoder_image(image_input)  # batch * num_channels_encoder
        input = torch.cat([start_input, image_features], dim=1)  # 设置初始输入,batch * input_size(self.embedding_size + self.num_channels_encoder)，融入图片特征
        last_h = start_h#设置初始状态
        for i in range(sequence_length):
            # 迭代GRU
            next_token, h, prediction = self.G_I2T(input, last_h, is_stochastic=False)#获得当前时间步预测的下一个token，隐藏状态和预测层,采用最大softmax方式抽样
            predictions.append(torch.unsqueeze(prediction, dim=0))
            input = torch.cat([self.embeddings(groundtrues[i]), image_features], dim=1)  # 输入正确的单词embedding,并融入图片特征
            last_h = h
        predictions = torch.cat(predictions, dim=0)
        return predictions#return seq_len, batch, vocab_size
    def generate_Y(self, x, seqlen):
        '''

        :param x: 输入的单词序列: batch * seq_len
        :param seqlen: 每条单词序列的实际长度:batch
        :return:返回生成的图像
        '''
        input_x = self.embeddings(x)  # batch * seq_len * embedding_size
        return self.G_T2I(input_x, seqlen)
    def pretraining(self, images, texts, seqlen):
        '''

        :param images: 对应的图像
        :param texts: 对应的文本
        :return:
        '''
        self.zero_grad()
        mse = nn.MSELoss()
        l1 = nn.L1Loss()
        # 利用文字信息生成图像
        y_gen = self.generate_Y(texts, seqlen)
        loss = mse(y_gen, images)
        loss.backward()
        self.T2I_pretraining_optimizer.step()
        return loss.item(), y_gen
    def forward(self, input): #  前向传播获得loss以及中间产物
        '''
        :param input[0]:batch * seq_len
        :param input[1]:batch * 3 * height * width
        :param input[2]:batch
        :return:loss_rsc_T2I:T2I路径重构的误差||loss_Gads_T2I:T2I路径中T2I生成器对抗的误差||loss_D_T2I:T2I路径判别器的误差||rsc_x:重构的句子||y_gen:利用x信息生成的图像
        '''
        x = input[0]
        y = input[1]
        seqlen = input[2]
        self.loss_rsc_T2I, self.loss_Gads_T2I, self.loss_D_T2I, self.x_rsc, self.y_gen = self.forward_T2I(x, y, seqlen)
        self.J_I2T, self.loss_D_I2T, self.J_rsc_I2T, self.loss_rsc_rewards_I2T, self.x_gen = self.forward_I2T(x, y, seqlen)
        return self.loss_rsc_T2I.item(), self.loss_Gads_T2I.item(), self.loss_D_T2I.item(), self.x_rsc, self.y_gen, self.J_I2T.item(), self.loss_D_I2T.item(), self.J_rsc_I2T.item(), self.loss_rsc_rewards_I2T.item(), self.x_gen
    def forward_T2I(self, x, y, seqlen):  # 文字生成图像的路线
        loss_func = nn.NLLLoss(ignore_index=self.padding_idx)
        mse = nn.MSELoss()
        # 利用文字信息生成图像
        y_gen = self.generate_Y(x, seqlen)
        x_trans = torch.transpose(x, dim0=0, dim1=1)
        # 利用图像信息重构文字
        if x.shape[0] ==self.batch_size:
            predictions = self.generate_pretrained(image_input=y_gen, start_input=self.start_input, start_h=self.start_h,
                                                     sequence_length=x.shape[1], groundtrues=x_trans)
        else:
            predictions = self.generate_pretrained(image_input=y_gen,
                                                   start_input=torch.zeros(x.shape[0], self.embedding_size).cuda(),
                                                   start_h=torch.zeros(x.shape[0], self.hidden_size_gru).cuda(),
                                                   sequence_length=x.shape[1], groundtrues=x_trans)
        # 获得重构误差
        loss_rsc = 0.0
        for t in range(x.shape[1]):
            loss_rsc += loss_func(torch.log(torch.clamp(predictions[t], min=1e-20, max=1.0)),
                              x_trans[t])  # tar*log(pre)
        loss_rsc = loss_rsc / x.shape[1]
        # 获得图像判别器误差
        pre_pos = self.D_T2I(y)
        pre_neg = self.D_T2I(y_gen.detach())  # 注意这里需要把y_gen从重构路径中分离
        loss_D = (mse(pre_pos, torch.ones(y.shape[0], 1).cuda()) + mse(pre_neg, torch.zeros(y.shape[0], 1).cuda())) / 2.0
        loss_D_l2 = 0.0
        loss_G_l2 = 0.0
        for param in self.D_T2I.output_layer.parameters():  # 加入L2正则化
            loss_D_l2 += torch.norm(param, p=2)
        for param in self.G_T2I.transform_layer.parameters():  # 加入L2正则化
            loss_G_l2 += torch.norm(param, p=2)
        loss_D += loss_D_l2 * 0.001
        # 获得T2I生成器的对抗误差
        pre_neg = self.D_T2I(y_gen)  # 注意这里不需要把y_gen从重构路径中分离
        loss_Gads = mse(pre_neg, torch.ones(y.shape[0], 1).cuda()) + loss_G_l2 * 0.001
        #展示重构后的输出
        x_rsc = torch.max(torch.transpose(predictions, dim0=0, dim1=1), dim=2)[1]  # batch, seq_len
        return loss_rsc, loss_Gads, loss_D, x_rsc, y_gen
    def onehot(self, label):
        a = torch.FloatTensor(label.shape[0], self.batch_size, self.vocab_size).zero_().cuda()
        return a.scatter_(dim=2, index=label, value=1)
    def calculate_realpolicy(self, policy_temp, record):  # 生成真正的策略policy，endtoken后的就不用计算了
        '''

        :param policy_temp: batch * seqlen
        :param record: dict
        :return: batch
        '''
        policy = []
        for i in range(policy_temp.shape[0]):
            policy.append(torch.prod(policy_temp[i:i+1, :record[i]+1], dim=1))
        policy = torch.unsqueeze(torch.cat(policy, dim=0), dim=1)  # batch * 1
        return policy
    def forward_I2T(self, x, y, seqlen):  # 图像生成文字的路线
        samples, hs, predictions, record, now_len = self.generate_X_nofixedlen(y, self.start_input, self.start_h, 20)  # 输入图片，生成句子
        trans_samples = torch.transpose(samples, dim0=0, dim1=1)  # batch * seq_len
        trans_samples_emb = self.embeddings(trans_samples)  # batch * seq_len * embedding_size
        seqlen_gen = [record[key] + 1 for key in range(x.shape[0])]  # 获得每一个生成句子的真实长度,用列表返回
        rewards = self.D_I2T(trans_samples_emb, seqlen_gen)  # 得到每一个生成句子的是否真实的奖励,一个生成句子一个奖励

        #策略梯度更新生成器的部分
        samples_onehot = self.onehot(label=torch.unsqueeze(samples, 2))  # 为每一个时刻的生成单词创建一个对应在字典的onehot位置，用于获取对应的策略p
        policy_temp = torch.transpose(torch.sum(samples_onehot * predictions, 2), dim0=0, dim1=1)  # 部分序列包含了endtoken后的策略：batch * seq_len
        policy = self.calculate_realpolicy(policy_temp=policy_temp, record=record)
        J = -(torch.sum(torch.log(torch.clamp(policy, min=1e-20, max=1.0)) * rewards) / x.shape[0])

        #更新I2T判别器的部分
        mse = nn.MSELoss()
        input_pos = self.embeddings(x)  # batch * seq_len * embedding_size
        pre_pos = self.D_I2T(input_pos, seqlen)
        pre_neg = rewards
        loss_D = (mse(pre_pos, torch.ones(x.shape[0], 1).cuda()) + mse(pre_neg, torch.zeros(x.shape[0], 1).cuda())) / 2.0

        #重构部分
        l1loss = nn.L1Loss(reduce=False)
        mse_rsc = nn.MSELoss(reduce=False)
        y_rsc = self.generate_Y(trans_samples, seqlen_gen)
        rsc_rewards = torch.sum(torch.sum(l1loss(y_rsc, y), dim=3), dim=2)/(self.width * self.height)
        rsc_rewards = 1.0 - torch.tanh(rsc_rewards)
        rsc_rewards_detach = rsc_rewards.detach()  # 把重构奖励当成权重
        loss_rsc_rewards = -torch.sum(rsc_rewards)/x.shape[0]  # 令奖励越大越好，用于更新生成器T2I
        print(rsc_rewards)
        J_rsc = -(torch.sum(torch.log(torch.clamp(policy, min=1e-20, max=1.0)) * rsc_rewards_detach) / x.shape[0])
        return J, loss_D, J_rsc, loss_rsc_rewards, trans_samples


    def backward(self, updateD = True):
        self.zero_grad()
        (self.loss_Gads_T2I + self.loss_rsc_T2I + self.J_I2T + self.J_rsc_I2T + self.loss_rsc_rewards_I2T * 0.1).backward(retain_graph=True)
        self.G_optimizer.step()
        if updateD:
            self.zero_grad()
            (self.loss_D_T2I + self.loss_D_I2T).backward()
            self.D_optimizer.step()




