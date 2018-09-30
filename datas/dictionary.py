# -*- coding:utf-8 -*-
class Dictionary_flickr30K():
    def __init__(self, src, start_idx, end_idx, padding_idx):
        texts = open(src).readlines()
        #应该把数据处理成doc*word的列表
        self.texts = []#存放原始数据
        self.idx2word = []
        self.word2idx = {}
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.padding_idx = padding_idx
        for text in texts:
            self.texts.append(text.split('\t')[1].replace('\n', '').lower().split(' '))
    def __len__(self):
        if self.idx2word.__len__()==self.word2idx.__len__():
            return self.idx2word.__len__()
        else:
            raise RuntimeError('dictionary size error')
    def build(self):
        temp = []#存放单词，唯一
        wordcount = []#存放单词以及它的频数
        for text in self.texts:
            for word in text:
                if word in temp:
                    idx = temp.index(word)#找出索引
                    if wordcount[idx][0]==temp[idx]:#计算频数
                        wordcount[idx][1] = wordcount[idx][1]+1
                    else:
                        raise RuntimeError('dictionary operation error')
                else:
                    temp.append(word)
                    wordcount.append([word, 0])

        wordcount.sort(key=lambda item: item[1], reverse=True)#按照频数排序
        self.idx2word.append('start token')#加入start token
        self.word2idx['start token'] = self.start_idx#为start token安排'0'idx
        self.idx2word.append('end token')#加入end token
        self.word2idx['end token'] = self.end_idx#为end token安排'1'idx
        self.idx2word.append('padding token')  # 加入padding token
        self.word2idx['padding token'] = self.padding_idx  # 为padding token安排'2'idx
        for idx in range(3, wordcount.__len__()+3):
            word = wordcount[idx-3][0]
            self.idx2word.append(word)
            self.word2idx[word] = idx
        print('Successfully building the dictionary !')