#!/usr/bin/env python
# coding: utf-8
""" 
# # 注意：
# 此文件已在colab完成训练生成grade.h模型文件和对应的参数文件parameter.kpl

# # 配置colab环境

# In[ ]:


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


# In[ ]:
"""

import os
import time
import random
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm
from operator import itemgetter
from collections import defaultdict
import torch
import jieba

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
else:
    device = torch.device('cpu')
print(device)
# 准备好模型的参数
parameter = {
    'epoch':100,
    'batch_size':300,
    'embedding_dim':300,
    'hidden_size':128,
    'num_layers':2, 
    'dropout':0.1,
    'cuda':torch.device('cuda'),
    #'cuda':torch.device('cpu'),
    'lr':0.001,
    'max_len':50,
}

def build_dataSet(parameter):
    data_src = pd.read_csv('./dataSet/data_src.csv')
    data_src = data_src[data_src['关系'] == 'question2answer'] #只取出关系类型是question2answer的
    q,a = list(data_src.实体1),list(data_src.实体2)
    word2id = defaultdict(int)
    word2id['<PAD>'] = 0
    word2id['<UNK>'] = 0
    qa_list = {}
    for ind in range(len(q)): #1390条
        q_cut = list(q[ind]) #每条question按word切分
        a_cut = list(a[ind]) #每条answer按word切分
        if q[ind] not in qa_list:
            qa_list[q[ind]] = [q_cut,a_cut] #qa_list:{'AutoML问题构成?': [['A', 'u', 't', 'o', 'M', 'L', '问', '题', '构', '成', '?'], ['特', '征', '选', '择']],...}
        else:
            qa_list[q[ind]] += [a_cut] #同一个问题的新的答案
        for i in q_cut:
            word2id[i] += 1
        for i in a_cut:
            word2id[i] += 1 #问题和答案放一起做embeding
            
    qa_list = list(qa_list.values())
    
    #每个元素由一个问题和多个对应的答案构成如:[['A', 'u', 't', 'o', 'M', 'L', '问', '题', '构', '成', '?'], ['特', '征', '选', '择'], ['模', '型', '选', '择'], ['算', '法', '选', '择']]
    parameter['qa_list'] = qa_list 
    parameter['word2id'] = dict(zip(word2id.keys(),range(len(word2id))))
    parameter['id2word'] = dict(zip(range(len(word2id)),word2id.keys()))
    parameter['word_size'] = len(word2id)
    
def sample(n,parameter,neg_sample_num):
    neg_sample = []
    q_size = len(parameter['qa_list'])#309
    while 1:
        sample_id = random.randint(0,q_size-1)
        if sample_id == n:
            continue
        neg_sample_answer = parameter['qa_list'][sample_id]
        a_id = random.randint(1,len(neg_sample_answer)-1) #产生用来抽取答案的随机数
        neg_sample.append(neg_sample_answer[a_id])
        if len(neg_sample) >= neg_sample_num:
            return neg_sample
        
def list2torch(a):
    return torch.from_numpy(np.array(a)).long().to(parameter['cuda'])
    
def batch_yield(parameter,shuffle = True):
    for train_epoch in range(parameter['epoch']):
        qa_list = parameter['qa_list']
        data = []
        for ind,i in enumerate(qa_list):
            q = i[0] #问题
            p_a = i[1:] #可能不止一个答案,p_a放正确的答案
            n_a = sample(ind,parameter,len(p_a)) #n_a放错误的答案，数目和p_a相同
            q = [q] * len(p_a)  
            data += list(zip(q,p_a,n_a)) #格式data[0]:[ ([q],[p_a],[n_a]), ([q],[p_a],[n_a]), ([q],[p_a],[n_a]) ]
        if shuffle:
            random.shuffle(data)
        batch_q,batch_a,batch_n = [],[],[]
        seq_len_q,seq_len_a,seq_len_n = 0,0,0
        for (q,a,n) in tqdm(data):
            q = itemgetter(*q)(parameter['word2id'])
            a = itemgetter(*a)(parameter['word2id'])
            n = itemgetter(*n)(parameter['word2id'])
            q = list(q) if type(q) == type(()) else [q,0]
            a = list(a) if type(a) == type(()) else [a,0]
            n = list(n) if type(n) == type(()) else [n,0]
            q = q[:parameter['max_len']]
            a = a[:parameter['max_len']]
            n = n[:parameter['max_len']]
            if len(q) > seq_len_q:
                seq_len_q = len(q)
            if len(a) > seq_len_a:
                seq_len_a = len(a)
            if len(n) > seq_len_n:
                seq_len_n = len(n)
            batch_q.append(q)
            batch_a.append(a)
            batch_n.append(n)
            if len(batch_q) >= parameter['batch_size']:
                batch_q = [i+[0]*(seq_len_q-len(i)) for i in batch_q]
                batch_a = [i+[0]*(seq_len_a-len(i)) for i in batch_a]
                batch_n = [i+[0]*(seq_len_n-len(i)) for i in batch_n]
                yield list2torch(batch_q),list2torch(batch_a),list2torch(batch_n),None,False
                batch_q,batch_a,batch_n = [],[],[]
                seq_len_q,seq_len_a,seq_len_n = 0,0,0
                
        batch_q = [i+[0]*(seq_len_q-len(i)) for i in batch_q]
        batch_a = [i+[0]*(seq_len_a-len(i)) for i in batch_a]
        batch_n = [i+[0]*(seq_len_n-len(i)) for i in batch_n]
        yield list2torch(batch_q),list2torch(batch_a),list2torch(batch_n),train_epoch,False
        
        batch_q,batch_a,batch_n = [],[],[]
        seq_len_q,seq_len_a,seq_len_n = 0,0,0
    yield None,None,None,None,True
    
#数据准备=========================
if not os.path.exists('parameter_tmp.pkl'):
    print('parameter_tmp.pkl not exist!')
    build_dataSet(parameter)
    pk.dump(parameter,open('parameter_tmp.pkl','wb'))
else:
    parameter=pk.load(open('parameter_tmp.pkl','rb'))
    print('parameter_tmp.pkl does exist!')


# In[ ]:


#若不生成parameter.pkl这里执行会报错
train_yield = batch_yield(parameter)
test_q,test_a,test_n,_,_ = next(train_yield)
test_q,test_a,test_n


# In[ ]:


import torch.nn.functional as F # pytorch 激活函数的类
from torch import nn,optim # 构建模型和优化器

# 构建分类模型
class TextRNN(nn.Module):
    def __init__(self, parameter):
        super(TextRNN, self).__init__()
        embedding_dim = parameter['embedding_dim']
        hidden_size = parameter['hidden_size']
        num_layers = parameter['num_layers']
        dropout = parameter['dropout']
        word_size = parameter['word_size']
        self.embedding = nn.Embedding(word_size, embedding_dim, padding_idx=0)
        
        self.lstm_q = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)

        self.lstm_a = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)


        
    def forward(self, q, a1,a2 = None):
        q_emd = self.embedding(q)
        q_emd,(h, c)= self.lstm_q(q_emd)
        q_emd = torch.max(q_emd,1)[0]

        a1_emd = self.embedding(a1)
        a1_emd,(h, c)= self.lstm_a(a1_emd)
        a1_emd = torch.max(a1_emd,1)[0]
        if a2 is not None:
            a2_emd = self.embedding(a2)
            a2_emd,(h, c)= self.lstm_a(a2_emd)
            a2_emd = torch.max(a2_emd,1)[0]
            return q_emd,a1_emd,a2_emd
        return F.cosine_similarity(q_emd,a1_emd,1,1e-8)



#若不生成parameter.pkl这里执行会报错
test_model = TextRNN(parameter).cuda()#我电脑不支持
#test_model = TextRNN(parameter)
test_model(test_q,test_a)




import os
import shutil
import pickle as pk
from torch.utils.tensorboard import SummaryWriter

# 构建模型
model = TextRNN(parameter).to(parameter['cuda'])

# 确定训练模式
model.train()

# 确定优化器和损失
optimizer = torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.95, nesterov=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

# 准备迭代器
train_yield = batch_yield(parameter)

# 开始训练
loss_cal = []
min_loss = float('inf')

if os.path.exists('grade.h5'):
    print("grade.h5 exist!")
else:
    print("grade.h5 not exist!")
    while 1:
        q,a,n,epoch,keys = next(train_yield)
        if keys:
            break
        q_emd,a_emd,n_emd = model(q,a,n)
        loss = nn.functional.triplet_margin_loss(q_emd, a_emd, n_emd,reduction='mean')#这是什么损失？？？-->三元损失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_cal.append(loss.item())
        if epoch is not None:
            if (epoch+1)%1 == 0:
                loss_cal = sum(loss_cal)/len(loss_cal)
                if loss_cal < min_loss:
                    min_loss = loss_cal
                    torch.save(model.state_dict(), 'grade.h5')
                print('epoch [{}/{}], Loss: {:.4f}'.format(epoch+1,                                                         parameter['epoch'],loss_cal))
                optimizer.step()
                loss_cal = [loss.item()]


# In[ ]:




