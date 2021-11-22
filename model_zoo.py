# Author: teddy
# Date: 2021-11-20 10:05:06
# LastEditTime: 2021-11-20 10:37:08
# LastEditors: Please set LastEditors
# Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
# FilePath: \undefinedf:\\firstlesson\myfirstlesson\第一次进行实体识别\model_zoo.py

import torch.nn.functional as F 
from torchcrf import CRF
from torch import nn 

# 构建基于bilstm实现ner
class bilstm(nn.Module):
    def __init__(self, parameter):
        super(bilstm, self).__init__()
        word_size = parameter['word_size']
        embedding_dim = parameter['d_model']
        self.embedding = nn.Embedding(word_size, embedding_dim, padding_idx=0)

        hidden_size = parameter['hid_dim']
        num_layers = parameter['n_layers']
        dropout = parameter['dropout']
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)

        output_size = parameter['output_size']
        self.fc = nn.Linear(hidden_size*2, output_size)
        
    def forward(self, x):
        out = self.embedding(x)
        out,(h, c)= self.lstm(out)
        out = self.fc(out)
        return out.view(-1,out.size(-1))

# 构建基于bilstm+crf实现ner
class bilstm_crf(nn.Module):
    def __init__(self, parameter):
        super(bilstm_crf, self).__init__()
        word_size = parameter['word_size']
        embedding_dim = parameter['d_model']
        self.embedding = nn.Embedding(word_size, embedding_dim, padding_idx=0)

        hidden_size = parameter['hid_dim']
        num_layers = parameter['n_layers']
        dropout = parameter['dropout']
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)

        output_size = parameter['output_size']
        self.fc = nn.Linear(hidden_size*2, output_size)
        
        self.crf = CRF(output_size,batch_first=True)
        
    def forward(self, x):
        out = self.embedding(x)
        out,(h, c)= self.lstm(out)
        out = self.fc(out)
        return out
    
