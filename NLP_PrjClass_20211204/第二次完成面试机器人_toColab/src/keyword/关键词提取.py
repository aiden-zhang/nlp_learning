#!/usr/bin/env python
# coding: utf-8

# # 注意：
# 此文件已在colab完成训练生成ber_crf.h5模型文件和对应的参数文件parameter.kpl

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
    if os.path.exists('/content/gdrive/MyDrive/第二次完成面试机器人_toColab/src/keyword'):
        return '/content/gdrive/MyDrive/第二次完成面试机器人_toColab/src/keyword/'
    else:
        return './' #在本地

# 3.调用系统命令，切换到对应工程路径，相当于cd，但是直接!cd是不行的
print("path:",get_root_dir())
os.chdir(get_root_dir())

# 4.再次确认路径
get_ipython().system('pwd')
get_ipython().system('ls')


# In[ ]:


from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm
import numpy as np
import random
import torch 
import jieba
import json
import os
import pickle as pk

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
else:
    device = torch.device('cpu')
print(device)


# In[ ]:


# 确定模型训练方式，GPU训练或CPU训练
parameter_copy = {
    # 此处embedding维度为768
    'd_model':768, 
    # rnn的隐层维度为300
    'hid_dim':300,
    # 训练的批次为100轮
    'epoch':20,
    # 单次训练的batch_size为100条数据
    'batch_size':50,
    # 设置序列的最大长度为100
    'n_layers':2,
    # 设置dropout，为防止过拟合
    'dropout':0.1,
    # 配置cpu、gpu
    'device':device,
    # 设置训练学习率
    'lr':0.001,
    # 优化器的参数，动量主要用于随机梯度下降
    'momentum':0.99,
    'max_len':50,
}

def build_dataSet(parameter,data_path = '../../dataSet/tagging.txt'):
    data = open(data_path,'r',encoding = 'utf-8').readlines()
    data_set = {'input':[],'label':[]}
    key_table = defaultdict(int)
    vocab_table = defaultdict(int)
    vocab_table['<PAD>'] = 0
    vocab_table['<UNK>'] = 0
    for i in data:
        i = i.strip().split()
        data_set['input'].append(i[0])
        data_set['label'].append(i[1])
        vocab_table[i[0]] += 1
        key_table[i[1]] += 1
    key2ind = dict(zip(key_table.keys(),range(len(key_table))))
    ind2key = dict(zip(range(len(key_table)),key_table.keys()))
    word2ind = dict(zip(vocab_table.keys(),range(len(vocab_table))))
    ind2word = dict(zip(range(len(vocab_table)),vocab_table.keys()))
    parameter['key2ind'] = key2ind
    parameter['ind2key'] = ind2key
    parameter['word2ind'] = word2ind
    parameter['ind2word'] = ind2word
    parameter['data_set'] = data_set
    parameter['output_size'] = len(key2ind)
    parameter['word_size'] = len(word2ind)
    return parameter

def sample(parameter):
    while 1:
        data_set = parameter['data_set']
        select_id = random.randint(0,len(data_set['label'])-parameter['max_len'])
        select_id = [select_id,select_id+parameter['max_len']-1]
        while data_set['label'][select_id[0]][0] not in ['O','B','S'] and select_id[0] < len(data_set['label']):
            select_id[0] += 1
        while data_set['label'][select_id[1]][0] not in ['O','E','S'] and select_id[1] > 0:
            select_id[1] -= 1
        if select_id[1] > select_id[0] and             data_set['label'][select_id[0]][0] in ['O','B','S'] and             data_set['label'][select_id[1]][0] in ['O','E','S']:
            select_label = data_set['label'][select_id[0]:select_id[1]+1]
            select_input = data_set['input'][select_id[0]:select_id[1]+1]
            return select_input,select_label
        else:
            continue


def batch_yield(parameter):
    Epoch = parameter['epoch'] 
    for epoch in range(Epoch):
        inputs,targets = [],[]
        max_len = 0
        for items in tqdm(range(10000)):
            input,label = sample(parameter)
            input = tokenizer.convert_tokens_to_ids(input)
            label = itemgetter(*label)(parameter['key2ind'])
            label = label if type(label) == type(()) else (label,0)
            if len(input) > max_len:
                max_len = len(input)
            inputs.append(list(input))
            targets.append(list(label))
            if len(inputs) >= parameter['batch_size']:
                inputs = [i+[0]*(max_len-len(i)) for i in inputs]
                targets = [i+[0]*(max_len-len(i)) for i in targets]
                if items < 10000-1:
                    yield list2torch(inputs),list2torch(targets),None,False
                else:
                    yield list2torch(inputs),list2torch(targets),epoch,False
                inputs,targets = [],[]
                max_len = 0
        inputs = [i+[0]*(max_len-len(i)) for i in inputs]
        targets = [i+[0]*(max_len-len(i)) for i in targets]
    yield None,None,None,True
            

def list2torch(ins):
    return torch.from_numpy(np.array(ins)).long().to(parameter['device'])

if os.path.exists('parameter.pkl'):
  print("parameter.pkl exist!")
else:
  print("parameter.pkl not exist!")
  parameter = build_dataSet(parameter_copy)
  pk.dump(parameter,open('parameter.pkl','wb'))


# In[ ]:


from transformers import WEIGHTS_NAME, BertConfig,get_linear_schedule_with_warmup,AdamW, BertTokenizer
from transformers import BertModel,BertPreTrainedModel
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch

import torch.nn.functional as F # pytorch 激活函数的类
from torch import nn,optim # 构建模型和优化器
from torchcrf import CRF

class bert_crf(BertPreTrainedModel):
    def __init__(self, config,parameter):
        super(bert_crf, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        embedding_dim = parameter['d_model']
        output_size = parameter['output_size']
        self.fc = nn.Linear(embedding_dim, output_size)
        self.init_weights()
        
        self.crf = CRF(output_size,batch_first=True)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,labels=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.fc(sequence_output)
        return logits
    
config_class, bert_crf, tokenizer_class = BertConfig, bert_crf, BertTokenizer
config = config_class.from_pretrained("prev_trained_model")
tokenizer = tokenizer_class.from_pretrained("prev_trained_model")


# In[ ]:


import os
import shutil
import pickle as pk
from torch.utils.tensorboard import SummaryWriter

random.seed(2019)

# 构建模型
model = bert_crf.from_pretrained("prev_trained_model",config=config,parameter = parameter).to(parameter['device'])

# 决定训练权重
full_finetuning = True
if full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0}
        ]
else: 
        param_optimizer = list(model.fc.named_parameters()) 
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]

# 确定优化器和策略
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, correct_bias=False)
train_steps_per_epoch = 10000 // parameter['batch_size']
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_steps_per_epoch, num_training_steps=parameter['epoch'] * train_steps_per_epoch)

# 确定训练模式
model.train()

# 准备迭代器
train_yield = batch_yield(parameter)

# 开始训练
loss_cal = []
min_loss = float('inf')
logging_steps = 0

if os.path.exists('bert_crf.h5'):
  print("bert_crf.h5 exist!")
else:
  print("bert_crf.h5 not exist!")
  while 1:
          inputs,targets,epoch,keys = next(train_yield)
          if keys:
              break
          out = model(inputs)
          loss = -model.crf(out,targets)
          optimizer.zero_grad()
          loss.backward()
          nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5)
          optimizer.step()
          scheduler.step()
          loss_cal.append(loss.item())
          logging_steps += 1
          if logging_steps%20 == 0:
              print(sum(loss_cal)/len(loss_cal))
          if epoch is not None:
              if (epoch+1)%1 == 0:
                  loss_cal = sum(loss_cal)/len(loss_cal)
                  if loss_cal < min_loss:
                      min_loss = loss_cal
                      torch.save(model.state_dict(), 'bert_crf.h5')
                  print('epoch [{}/{}], Loss: {:.4f}'.format(epoch+1,                                                         parameter['epoch'],loss_cal))
              loss_cal = [loss.item()]


# In[ ]:




