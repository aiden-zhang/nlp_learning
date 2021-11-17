#coding:utf-8

import jieba
import pandas as pd
from tqdm import tqdm

def make_one_hot(data,voc_size):
    return (np.arange(voc_size)==np.array(data)[:,None]).astype(np.integer)

def build_vocab(corpus):
    vocab = defaultdict(int)
    for i in corpus:
        for j in i:
            vocab[j] += 1
    vocab_size = len(vocab)
    voc_index = dict(zip(vocab.keys(),range(len(vocab.keys()))))
    index_voc = dict(zip(range(len(vocab.keys())),vocab.keys()))
    return voc_index,index_voc,vocab_size
    
def get_train_data(corpus,voc_index,vocab_size):
    for train_epoch in range(parameter['epochs']):
        print('now is '+str(train_epoch)+' epoch')
        for sentence in tqdm(corpus):
            this_len = len(sentence)
            for ind in range(this_len):
                context = sentence[max(0,ind-parameter['window']):ind]+sentence[ind+1:min(this_len,ind+parameter['window']+1)]
                context = make_one_hot(itemgetter(*context)(voc_index),vocab_size)
                word = make_one_hot([voc_index[sentence[ind]]],vocab_size)
                yield context,word,True
    yield None,None,False
#             train_data.append([context,word])
#     return train_data,voc_index,index_voc


if __name__== "__main__":
    
    # 1.数据准备######################################################
    data_chinese = []
    # data_chinese = pd.read_csv('data/诗句.csv')['content']
    data_math = [i[0] for i in pd.read_csv('data/数学原始数据.csv',encoding = 'gbk').values]
    data = list(data_chinese) + list(data_math)
    print(len(data_chinese),len(data_math),len(data))
    del data_chinese,data_math
    
    stop_words = '。，、（）().,:：\\、"“”；;||？?^<>シ[]·【】け√《》°{}\u3000'#abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890+-/
    data_after_cut = []
    for i in tqdm(data):
        cut_sen = [j for j in jieba.lcut(i) if j not in stop_words]
        data_after_cut.append(cut_sen)    
    print(data_after_cut[0:5])
    
    # 2.数据预处理################################################################################################
    parameter = {
        'vector_size':50, #隐层节点数或词向量大小
        'window':4,  #ngram大小
        'alpha':0.01, #学习率
        'epochs':4  #学习批次
    }
    from collections import defaultdict
    from operator import itemgetter
    from tqdm import tqdm
    import numpy as np
    
    voc_to_index,index_to_voc,vocab_size = build_vocab(data_after_cut)
    
    #初始化一个迭代器
    train_data = get_train_data(data_after_cut,voc_to_index,vocab_size) 
    
    # #have a look
    for i in range(1):
        context,word,keys = next(train_data) #可得到中心词word::1x5260和上下文context::4x5260的onehot表示
        print(context)
    # context,word,keys
    
    # 3.创建模型############################################################################
    import pickle as pk
    
    class word2vec():
        def __init__(self,parameter,voc_index,index_voc):
            self.alpha = parameter['alpha']
            self.vector_size = parameter['vector_size']
            self.w1,self.w2 = None,None #
            self.vocab_size = len(voc_index)
            self.voc_index = voc_index
            self.index_voc = index_voc
            self.init_weight()
            
        def init_weight(self):
            self.w1 = np.random.uniform(-1, 1, (self.vocab_size, self.vector_size))
            self.w2 = np.random.uniform(-1, 1, (self.vector_size, self.vocab_size))
            
        def load_weight(self,file_name):
            pass
        
        def forward(self,x):
            h = np.dot(x, self.w1)
            u = np.dot(h, self.w2)
            y_predict = self.softmax(u)
            return y_predict,h,u
        
        def softmax(self,x):
            e = np.exp(x)
            return e/e.sum()
        
        def backward(self, e, h, x):
            dw2 = np.outer(h, e)
            dw1 = np.outer(x, np.dot(self.w2, e.T))
            self.w1 = self.w1 - (self.alpha * dw1)
            self.w2 = self.w2 - (self.alpha * dw2)
    
        def save(self):
            pk.dump([self.w1,self.voc_index,self.index_voc],open('my_first_word2vec_model.pkl','wb'))
            
        def train(self,data):#data为数据的迭代器
            while 1:
                context,word,keys = next(data)
                if not keys:
                    break
                y_predict,h,u = self.forward(word)
                EI = np.sum([np.subtract(y_predict, y) for y in context], axis=0)
                self.backward(EI, h, word)
            self.save()
            
    model = word2vec(parameter,voc_to_index,index_to_voc)    