#!/usr/bin/env python
# coding: utf-8

# #注意:
# 此文件已在wingPro验证，通过cpu推理

""" 
# #配置colab环境

#colab中运行jupyter文件的步骤：
# 1.挂载云盘
from google.colab import drive
drive.mount('/content/gdrive')

# 2.安装需要的软件
get_ipython().system('pip3 install transformers')
get_ipython().system('pip3 install pytorch-crf')

import os
def get_root_dir():
    if os.path.exists('/content/gdrive/MyDrive/第二次完成面试机器人_toColab/src/grade'):
        return '/content/gdrive/MyDrive/第二次完成面试机器人_toColab/src/grade/'
    else:
        return './' #在本地

# 3.调用系统命令，切换到对应工程路径，相当于cd，但是直接!cd是不行的
print("path:",get_root_dir())
os.chdir(get_root_dir())

# 4.再次确认路径
get_ipython().system('pwd')
get_ipython().system('ls')

"""

import torch.nn.functional as F # pytorch 激活函数的类
from torch import nn,optim # 构建模型和优化器
import torch
import pickle as pk
import numpy as np
import jieba
import pdb

class Grade(nn.Module):
    def __init__(self, parameter):
        super(Grade, self).__init__()
        embedding_dim = parameter['embedding_dim']
        hidden_size = parameter['hidden_size']
        num_layers = parameter['num_layers']
        dropout = parameter['dropout']
        word_size = parameter['word_size']
        self.embedding = nn.Embedding(word_size, embedding_dim, padding_idx=0)
        
        self.lstm_q = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)

        self.lstm_a = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)

        
    def forward(self, q, a1,a2 = None):
        q_emd = self.embedding(q)
        q_emd,(h, c)= self.lstm_q(q_emd) #1x9x256
        q_emd = torch.max(q_emd,1)[0] #1x256

        a1_emd = self.embedding(a1)
        a1_emd,(h, c)= self.lstm_a(a1_emd)
        a1_emd = torch.max(a1_emd,1)[0]
        if a2 is not None:
            a2_emd = self.embedding(a2)
            a2_emd,(h, c)= self.lstm_a(a2_emd)
            a2_emd = torch.max(a2_emd,1)[0]
            return q_emd,a1_emd,a2_emd
        
        #q a经过lstm特征提取器得到的隐层的输出-》求两个相量的相似性
        return F.cosine_similarity(q_emd,a1_emd,1,1e-8)


    
def list2torch(a):
    return torch.from_numpy(np.array(a)).long().to(parameter['cuda'])

def predict(model,parameter,q,a):
    #pdb.set_trace()
    q = list(q)
    a = list(a)
    q_cut = []
    for i in q:
        if i in parameter['word2id']:
            q_cut.append(parameter['word2id'][i])
        else:
            q_cut.append(parameter['word2id']['<UNK>'])
    a_cut = []
    for i in a:
        if i in parameter['word2id']:
            a_cut.append(parameter['word2id'][i])
        else:
            a_cut.append(parameter['word2id']['<UNK>'])
    print(q_cut,a_cut)
    q_cut,a_cut = [q_cut[:parameter['max_len']]],[a_cut[:parameter['max_len']]]
    prob = model(list2torch(q_cut),list2torch(a_cut))
    print(prob)
    return prob.cpu().item()

def load_model(root_path = './'):
    parameter = pk.load(open(root_path+'parameter_grade.pkl','rb'))
    print(parameter)
    parameter['cuda']='cpu' #在cpu上测试推理
    model = Grade(parameter).to(parameter['cuda'])
    
    #model.load_state_dict(torch.load(root_path+'grade.h5'))
    model.load_state_dict(torch.load(root_path+'grade.h5',map_location='cpu'))#模型映射到cpu
    model.eval()
    return model,parameter

if __name__== "__main__":
    
    model,parameter = load_model()
    #x=torch.rand(100,35,300)
    #y=torch.rand(100,32,300)
    #ret = model(x,y)
    q = '特征工程选择思路？'
    a = '基于统计信息的，熵、相关性、KL系数'
    prob = predict(model,parameter,q,a)
    prob

