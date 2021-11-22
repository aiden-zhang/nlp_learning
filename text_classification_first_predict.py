#coding:utf-8
# 来源：整理自深兰NLP项目课:意图识别推理(最新版) -->已完成
# 功能：使用先前训练好的模型，评判模型结果
# 备注：只实现了textRNN，是双向lstm; 有空可以实现TextCNN和TextRCNN

import torch
import numpy as np
import pandas as pd
import pickle as pk
from tqdm import tqdm
import torch.nn.functional as F # pytorch 激活函数的类
from torch import nn,optim # 构建模型和优化器
from operator import itemgetter
from collections import defaultdict


# 构建分类模型
class TextRNN(nn.Module):
    def __init__(self, parameter):
        super(TextRNN, self).__init__()
        embedding_dim = parameter['embedding_dim']
        hidden_size = parameter['hidden_size']
        output_size = parameter['output_size']
        num_layers = parameter['num_layers']
        dropout = parameter['dropout']
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size*2, output_size)
        
    def forward(self, x):
        out,(h, c)= self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def batch_yield(chars,labels,parameter,shuffle = True):
    for train_epoch in range(parameter['epoch']):
        if shuffle:
            permutation = np.random.permutation(len(chars))
            chars = chars[permutation]
            labels = labels[permutation]
        max_len = 0
        batch_x,batch_y = [],[]
        for iters in tqdm(range(len(chars))):
            batch_ids = itemgetter(*chars[iters])(parameter['char2ind'])
            try:
                batch_ids = list(batch_ids)
            except:
                batch_ids = [batch_ids,0]
            if len(batch_ids) > max_len:
                max_len = len(batch_ids)
            batch_x.append(batch_ids)
            batch_y.append(labels[iters])
            if len(batch_x) >= parameter['batch_size']:
                batch_x = [np.array(list(itemgetter(*x_ids)(parameter['ind2embeding']))+[parameter['ind2embeding'][0]]*(max_len-len(x_ids))) for x_ids in batch_x]
                device = parameter['cuda']
                yield torch.from_numpy(np.array(batch_x)).to(device),np.array(batch_y),True,None
                max_len,batch_x,batch_y = 0,[],[]
        batch_x = [np.array(list(itemgetter(*x_ids)(parameter['ind2embeding']))+[parameter['ind2embeding'][0]]*(max_len-len(x_ids))) for x_ids in batch_x]
        device = parameter['cuda']
        yield torch.from_numpy(np.array(batch_x)).to(device),np.array(batch_y),True,train_epoch
        max_len,batch_x,batch_y = 0,[],[]
    yield None,None,False,None
    
def load_model(way = 'TextRNN'):
    [train_chars,test_chars,train_labels,test_labels] = pk.load(open('dataSet.pkl','rb'))
    parameter = pk.load(open('parameter.pkl','rb'))
#     parameter['cuda'] = torch.device('cpu')
    parameter['dropout'] = 0
    model = eval(way+"(parameter).to(parameter['cuda'])")
    if way == 'TextRNN':
        model.load_state_dict(torch.load('model-rnn.h5'))
    if way == 'TextCNN':
        model.load_state_dict(torch.load('model-cnn.h5'))
    if way == 'TextRCNN':
        model.load_state_dict(torch.load('model-rcnn.h5'))
    return parameter,model,[test_chars,test_labels]

def compare(real,predict,histroy,parameter):
    com = real - predict
    for i in range(parameter['output_size']):
        histroy[i]['tp'] += len(np.where((com == 0) & (real == i))[0])
        histroy[i]['all_real'] += len(np.where((real == i))[0])
        histroy[i]['all_predict'] += len(np.where((predict == i))[0])

def toEstimate(way = 'TextRNN'):
    parameter,model,[test_chars,test_labels] = load_model(way)
    model.eval() 
    parameter['epoch'] = 1
    test_yield = batch_yield(test_chars,test_labels,parameter)
    histroy = dict(zip(range(parameter['output_size']),[{'tp':0,'all_real':0,'all_predict':0} for i in range(parameter['output_size'])]))
    
    while 1:
        seqs,labels,keys,epoch = next(test_yield)
        if not keys:
            break
        res = model(seqs)
        predicted_prob,predicted_index = torch.max(F.softmax(res, 1), 1)
        res = predicted_index.cpu().numpy()
        compare(labels,res,histroy,parameter)
    tp,all_real,all_predict = [histroy[i]['tp'] for i in histroy],[histroy[i]['all_real'] for i in histroy],\
    [histroy[i]['all_predict'] for i in histroy]
    tp.append(sum(tp))
    all_real.append(sum(all_real))
    all_predict.append(sum(all_predict))
    res = pd.DataFrame(np.array([tp,all_real,all_predict]).transpose())
    res.columns = ['tp','all_real','all_predict']
    res['recall'] = res['tp']/res['all_real']
    res['precision'] = res['tp']/res['all_predict']
    res['f1'] = 2*res['recall']*res['precision']/(res['recall']+res['precision'])
    res.index =['标签'+str(i) for i in range(15)]+['综合']
    return res

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


if __name__== "__main__":
    
    toEstimate('TextRNN')