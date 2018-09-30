# -*- coding:utf-8 -*-
import os
import random

from torch.utils.data import Dataset
from PIL import Image
from datas.dictionary import Dictionary_flickr30K
class Dataset_flickr30K(Dataset):
    def __init__(self, images_src, texts_src, transform, start_idx, end_idx, padding_idx):
        '''

        :param images_src: 图片文件夹路径
        :param texts_src: 文本文件路径
        :param transform: 转换
        '''
        self.images_src = images_src
        self.images_name = os.listdir(images_src)  # 图片的名称
        self.transform = transform
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.padding_idx = padding_idx
        self.dictionary = Dictionary_flickr30K(texts_src, start_idx, end_idx, padding_idx)
        self.dictionary.build()
        self.texts = [text.split('\t')[1].replace('\n', '').lower().split(' ')for text in open(texts_src, 'r').readlines()]  # 读取成doc * word的list
        self.texts = self.transform_word2idx(self.texts)  # 把数据转换成idx形式
        self.maxseqlen = self.get_maxseqlen()  # 最大的序列（句子+start_token+end_token）长度
    def __len__(self):
        return self.texts.__len__()
    def get_maxseqlen(self):  # 返回最大的序列长度,包含start_token和end_token
        return max([text.__len__() for text in self.texts])
    def transform_word2idx(self, texts):  # 把texts转换成idx形式,并且插入start_idx和end_idx
        for text in texts:
            for i in range(text.__len__()):
                text[i] = self.dictionary.word2idx[text[i]]
            text.insert(0, self.start_idx)
            text.append(self.end_idx)
        return texts
    def show(self, data):
        #展现数据的原始样貌
        result = []
        for doc in data:
            result.append([self.dictionary.idx2word[word] for word in doc])
        return result
    def __getitem__(self, item):
        image = Image.open(self.images_src+self.images_name[random.randint(0, self.images_name.__len__()-1)]).convert('RGB')
        image = self.transform(image)
        text = self.texts[item]
        return image, text
class Dataset_SketchyScene(Dataset):
    def __init__(self, images_src, texts_src, transform, start_idx, end_idx, padding_idx):
        '''

        :param images_src: 图片文件夹路径
        :param texts_src: 文本文件路径
        :param transform: 转换
        '''
        self.images_src = images_src
        self.images_name = os.listdir(images_src)  # 图片的名称
        self.transform = transform
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.padding_idx = padding_idx
        self.dictionary = Dictionary_flickr30K(texts_src, start_idx, end_idx, padding_idx)
        self.dictionary.build()
        self.texts = [text.split('\t')[1].replace('\n', '').lower().split(' ')for text in open(texts_src, 'r').readlines()]  # 读取成doc * word的list
        self.texts = self.transform_word2idx(self.texts)  # 把数据转换成idx形式
        self.maxseqlen = self.get_maxseqlen()  # 最大的序列（句子+start_token+end_token）长度
    def __len__(self):
        return self.texts.__len__()
    def get_maxseqlen(self):  # 返回最大的序列长度,包含start_token和end_token
        return max([text.__len__() for text in self.texts])
    def transform_word2idx(self, texts):  # 把texts转换成idx形式,并且插入start_idx和end_idx
        for text in texts:
            for i in range(text.__len__()):
                text[i] = self.dictionary.word2idx[text[i]]
            text.insert(0, self.start_idx)
            text.append(self.end_idx)
        return texts
    def show(self, data):
        #展现数据的原始样貌
        result = []
        for doc in data:
            result.append([self.dictionary.idx2word[word] for word in doc])
        return result
    def __getitem__(self, item):
        image = Image.open(self.images_src+self.images_name[random.randint(0, self.images_name.__len__()-1)]).convert('L')  # 灰度模式
        image = self.transform(image)
        text = self.texts[item]
        return image, text