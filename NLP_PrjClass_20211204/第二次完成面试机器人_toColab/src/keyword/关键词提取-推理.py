#!/usr/bin/env python
# coding: utf-8

# # 注意:
# 此代码已经过colab验证，其实就是ner模型

# 配置colab环境

# In[1]:


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


# In[4]:


from transformers import WEIGHTS_NAME, BertConfig,get_linear_schedule_with_warmup,AdamW, BertTokenizer
from transformers import BertModel,BertPreTrainedModel
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch
from operator import itemgetter
import torch.nn.functional as F # pytorch 激活函数的类
from torch import nn,optim # 构建模型和优化器
from torchcrf import CRF
import pickle as pk
import numpy as np

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


def load_model(root_path = './'):
    parameter = pk.load(open(root_path+'parameter.pkl','rb'))
    model = bert_crf(config,parameter).to(parameter['device'])
    model.load_state_dict(torch.load(root_path+'bert_crf.h5'))
    model.eval()
    return model,parameter

def list2torch(ins):
    return torch.from_numpy(np.array(ins)).long().to(parameter['device'])

def keyword_predict(input):
    input = list(input)
    input_id = tokenizer.convert_tokens_to_ids(input)
    predict = model.crf.decode(model(list2torch([input_id])))[0]
    predict = itemgetter(*predict)(parameter['ind2key'])
    keys_list = []
    for ind,i in enumerate(predict):
        if i == 'O':
            continue
        if i[0] == 'S':
            if not(len(keys_list) == 0 or keys_list[-1][-1]):
                del keys_list[-1]
            keys_list.append([input[ind],[i],[ind],True])
            continue
        if i[0] == 'B':
            if not(len(keys_list) == 0 or keys_list[-1][-1]):
                del keys_list[-1]
            keys_list.append([input[ind],[i],[ind],False])
            continue
        if i[0] == 'I':
            if len(keys_list) > 0 and not keys_list[-1][-1] and             keys_list[-1][1][0].split('-')[1] == i.split('-')[1]:
                keys_list[-1][0] += input[ind]
                keys_list[-1][1] += [i]
                keys_list[-1][2] += [ind]
            else:
                del keys_list[-1]
            continue
        if i[0] == 'E':
            if len(keys_list) > 0 and not keys_list[-1][-1] and             keys_list[-1][1][0].split('-')[1] == i.split('-')[1]:
                keys_list[-1][0] += input[ind]
                keys_list[-1][1] += [i]
                keys_list[-1][2] += [ind]
                keys_list[-1][3] = True
            else:
                del keys_list[-1]
            continue
#     print(keys_list)
    keys_list = [i[0] for i in keys_list]
    return keys_list

model,parameter = load_model()


# In[5]:


test_sentence = '如果明明知道隐变量在此次分类的过程中起到非常巨大作用的情况下，判别模型对隐变量的学习往往通过人为构造，更加不确定性'
res = keyword_predict(test_sentence)
res


# In[ ]:




