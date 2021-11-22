#coding:utf-8
# 来源：整理自深兰NLP项目课:意图识别推理(最新版) -->已完成
# 功能：使用先前训练好的模型推理给定句子是15个分类中的哪一个
# 备注：只实现了textRNN，是双向lstm; 有空可以实现TextCNN和TextRCNN

import torch.nn.functional as F # pytorch 激活函数的类
from torch import nn,optim # 构建模型和优化器
import pickle as pk
import numpy as np
import torch

def batch_yield_predict(chars,parameter):
    batch_x,batch_y = [],[]
    for iters in range(len(chars)):
        if chars[iters] in parameter['char2ind']:
            batch_x.append(parameter['ind2embeding'][parameter['char2ind'][chars[iters]]])
        else:
            batch_x.append(parameter['ind2embeding'][parameter['char2ind']['<unk>']])
    batch_x = [batch_x]
#     batch_y = [0]
    device = parameter['cuda']
    return torch.from_numpy(np.array(batch_x)).to(device)#,torch.from_numpy(np.array(batch_y)).to(device).long()

class TextRCNN(nn.Module):
    def __init__(self, parameter):
        super(TextRCNN, self).__init__()
        embedding_dim = parameter['embedding_dim']
        hidden_size = parameter['hidden_size']
        output_size = parameter['output_size']
        num_layers = parameter['num_layers']
        dropout = parameter['dropout']
        self.lstm = nn.LSTM(embedding_dim,hidden_size, \
                            num_layers, bidirectional=True, \
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2, output_size)
        self.fc_for_concat = nn.Linear(hidden_size * 2 + embedding_dim, hidden_size * 2)
    
    def forward(self, x):
        out,(h, c)= self.lstm(x)
        out = self.fc_for_concat(torch.cat((x, out), 2))
        # 激活函数
        out = F.sigmoid(out)
        out = out.permute(0, 2, 1)
        try:
            out = F.max_pool1d(out, out.size(2).item())
        except:
            out = F.max_pool1d(out, out.size(2))
        out = out.squeeze(-1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def load_model():
    parameter = pk.load(open('parameter.pkl','rb'))
    model = TextRCNN(parameter).to(parameter['cuda'])
    model.load_state_dict(torch.load('model-rcnn.h5'))
    return parameter,model

def predict(model,parameter,strs):
    strs = strs.split()
    strs = batch_yield_predict(strs,parameter)
    outputs = model(strs)
    predicted_prob,predicted_index = torch.max(F.softmax(outputs, 1), 1)
    return predicted_index.item(),predicted_prob.item()
    

parameter,model = load_model()

test = 'nnt 演 过 那 些 戏'
predict(model,parameter,test)
