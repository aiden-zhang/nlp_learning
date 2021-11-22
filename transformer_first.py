
#coding:utf-8
ins = []
outs = []
num = 1000
with open('train.zh','r',encoding = 'utf-8') as f:
    for ind,i in enumerate(f):
        # 适当数据清洗
        i = [tmp for tmp in i.strip() if tmp not in '？，。！ ']
        ins.append(i)
        # 用于快速学习模型，故此处选择1000条数据进行训练
        if ind > num:
            break
            
with open('train.en','r',encoding = 'utf-8') as f:
    for ind,i in enumerate(f):
        # 适当数据清洗
        i = [tmp for tmp in i.strip().strip('.').strip('?').strip(',').split()]
        outs.append(i)
        if ind > num:
            break
            
            
del f,i,ind,num

#基于本地的服务

# bert_client.encode(['<pad>','like']).shape

from collections import defaultdict
from operator import itemgetter
import torch.utils.data as Data
from tqdm import tqdm
import torch.nn as nn
import pickle as pk
import numpy as np
import torch
import math
import os

import time
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
else:
    device = torch.device('cpu')
parameter_copy = {
    # 为了加快模型收敛速度，此处调用基于bert模型得到的词向量，所用词向量的维度为768维
    'd_model':768, # Embedding Size
    'epoch':50,
    'alpha':0.01,
    'batch_size':5,
    # 设置序列的最大长度为100
    'max_len':100,
    'd_k':64, #k,q,v的隐层节点数
    'd_q':64,
    'd_v':64,
    'd_ff':2048, #ffn的隐层节点数
    'n_heads':1,
    'n_layers':6,
    # 设置最低频率的词
    'min_count':1,
    # 设置dropout，为防止过拟合
    'dropout':0.1,
    # 配置cpu、gpu
    'device':device,
    # 设置训练学习率
    'lr':0.001,
    # 优化器的参数，动量主要用于随机梯度下降
    'momentum':0.99,
}

def build_vocab(parameter,corpus):
    '''
    parameter:预设的配置项，
    corpus：训练的语料
    本函数的主要作用就是构建字典，包括有编码层的字典及解码层的字典，
    具体来说就是，本模型是数据中文语料进行翻译得到英文语料，即编码层是对中文语料的数据进行编码，解码得到英文语料的结果
    同时，本函数在处理的过程中会去除频率较低的词汇，以及调用基于bert训练好的词向量（对中文英文语料进行编码）
    '''
    global bert_client
    # 准备好用于统计词频的变量
    vocab = defaultdict(int)
    # 配置标志位，包括有补齐位:<PAD>，用于针对不定长的序列进行补齐；未知位:<UNK>，用于推理阶段得到未在字典中出现词汇的异常处理；
    # 起始位:<STA>，结束位:<END>，编解码系列模型的特色，应用于训练阶段于推理阶段，其作用对于编解码系列模型有极为重要的意义
    vocab['<PAD>'] = parameter['min_count']
    vocab['<UNK>'] = parameter['min_count']
    vocab['<STA>'] = parameter['min_count']
    vocab['<END>'] = parameter['min_count']
    # 统计词频
    for i in corpus:
        for j in i:
            vocab[j] += 1
    # 去除低频词
    for i in vocab:
        if vocab[i] < parameter['min_count']:
            del vocab[i]
    # 获取字典大小
    vocab_size = len(vocab)
    # 构建基于词-》id的字典
    voc_index = dict(zip(vocab.keys(),range(len(vocab.keys()))))
    # 构建基于id-》词的字典
    index_voc = dict(zip(range(len(vocab.keys())),vocab.keys()))
    # 调用bert训练好的词向量，若不想使用bert训练好的词向量，此处可替换为onehot形式的编码，模型侧需要添加编码层
    emb_list = bert_client.encode(list(voc_index.keys()))
    return voc_index,index_voc,vocab_size,emb_list

def batch_yield(parameter,ins,outs,shuffle = True):
    '''
    本函数的主要作用是亲手构建一个训练数据的迭代器，功能包括有：
        1、准备好面向于编码层的输入，即模型入参；
        2、准备面向于解码层，用于teacherforce的输入；
        3、准备好面向于解码层的输出，即模型的target（此处的输入输出和teacherforce的输入面向于编解码结构）
        4、在准备好输入输出及用于teacherforce的解码层输入后，对其进行补齐处理，处理成定长
    '''
    device = parameter['device']
    # 此处是一个迭代器，故第一个循环是训练的总批次，epoch
    for epoch in range(parameter['epoch']):
        # 每轮对原始数据进行随机化
        if shuffle:
            permutation = np.random.permutation(len(ins))
            ins = ins[permutation]
            outs = outs[permutation]
        # 准备好一个batch相应的lis（输入输出和teacherforce的输入）
        enc_input_list = []
        dec_input_list = []
        dec_output_list = []
        for items in tqdm(range(len(ins))):
            # 通过语料到id的字典，使用itemgetter批量对其进行处理，最终结果得到基于id的序列
            ids = itemgetter(*ins[items])(parameter['input_voc_index'])
            # 此处是对异常数据进行处理，出现的原因是使用itemgetter时，若序列长度为1则直接返回id，若序列长度大于1则返回元组
            # 为方便后面的处理对直接返回id的情形，替换为补了一个pad的元组
            ids = ids if type(ids) == type(()) else (ids,0)
            # 元组替换为数组，值得注意的是原文中对于编码层的输入也添加了sos，eos，事实上可以不使用
            enc_input_list.append(list(ids))
            # 进行相同的梳理，得到相应基于id的序列
            ids = itemgetter(*outs[items])(parameter['taget_voc_index'])
            # 值得注意的是，这边是必须使用sos和eos，因为训练阶段用于teacherforce的输入和输出必须要错位才可以进行训练，
            # 因此需要sos和eos进行区分错位，而推理阶段，获得编码层结果后进行解码时需要sos起始位开始解码，eos结束位结束推理
            if type(ids) == type(()):
                dec_input_list.append([parameter['taget_voc_index']['<STA>']]+list(ids))
                dec_output_list.append(list(ids)+[parameter['taget_voc_index']['<END>']])
            else:
                dec_input_list.append([parameter['taget_voc_index']['<STA>']]+[ids])
                dec_output_list.append([ids]+[parameter['taget_voc_index']['<END>']])
            # 当该个list大小达到batch_size大小则需要返回相应的数据
            if len(dec_output_list) >= parameter['batch_size']:
                # 先计算输出输出和teacherforce的输入序列的大小
                enc_input_len_list = [len(i) for i in enc_input_list]
                dec_input_len_list = [len(i) for i in dec_input_list]
                dec_output_len_list = [len(i) for i in dec_output_list]
                # 将对应id号替换为提前准备好的对应id号词的词向量
                # 值得注意：我们在使用build_vocab这个函数过程中准备好了词到id的字典和id到词的字典，以及基于id可以得到对应词的词向量
                # 前面的操作中已经把词转换为id，变成了id的序列，这一步工作就是基于id得到对应词的词向量，词向量的序列
                # 同时基于各项序列大小的数组，确定最大的序列长度，对其进行补齐
                enc_input_list = [parameter['input_emb'][i+[0]*(max(enc_input_len_list)-len(i))] for i in enc_input_list]
                dec_input_list = [parameter['target_emb'][i+[0]*(max(dec_input_len_list)-len(i))] for i in dec_input_list]
                dec_output_list = [i+[0]*(max(dec_output_len_list)-len(i)) for i in dec_output_list]
                # 迭代器返回，编码器输入，用于teacherforce的输入，解码器的输出，编码器输入的长度，teacherforce输入的长度，解码器的输入，当前的epoch，是否结束训练的标志
                yield torch.from_numpy(np.array(enc_input_list)).to(device),torch.from_numpy(np.array(dec_input_list)).to(device),torch.from_numpy(np.array(dec_output_list)).to(device).long(),enc_input_len_list,dec_input_len_list,dec_output_len_list,None,True
                # 数据返回后记录清空，重新开始提取数据
                enc_input_list,dec_input_list,dec_output_list = [],[],[]
        # 当前轮的最后，处理方式和上述一致
        enc_input_len_list = [len(i) for i in enc_input_list]
        dec_input_len_list = [len(i) for i in dec_input_list]
        dec_output_len_list = [len(i) for i in dec_output_list]
        enc_input_list = [parameter['input_emb'][i+[0]*(max(enc_input_len_list)-len(i))] for i in enc_input_list]
        dec_input_list = [parameter['target_emb'][i+[0]*(max(dec_input_len_list)-len(i))] for i in dec_input_list]
        dec_output_list = [i+[0]*(max(dec_output_len_list)-len(i)) for i in dec_output_list]
        yield torch.from_numpy(np.array(enc_input_list)).to(device),torch.from_numpy(np.array(dec_input_list)).to(device),torch.from_numpy(np.array(dec_output_list)).to(device).long(),enc_input_len_list,dec_input_len_list,dec_output_len_list,epoch,True
    # 完成所有轮数据的提取
    yield None,None,None,None,None,None,None,False

# 因为考虑到部分同学没有配置BertClient，无法直接调用bert训练好的词向量，因此这边提前配置好用于训练的相关参数
# 不要每次调用bert
if not os.path.exists('parameter.pkl'):
    from bert_serving.client import BertClient
    bert_client = BertClient()
    parameter = parameter_copy
    # 构建相关字典和对应的基于bert的词向量
    input_voc_index,input_index_voc,input_vocab_size,input_emb = build_vocab(parameter,ins)
    taget_voc_index,taget_index_voc,taget_vocab_size,target_emb = build_vocab(parameter,outs)
    # 将所获取的字典及对应词向量放置于parameter中，所有过程统一使用parameter进行处理
    parameter['input_voc_index'] = input_voc_index
    parameter['input_index_voc'] = input_index_voc
    parameter['taget_voc_index'] = taget_voc_index
    parameter['taget_index_voc'] = taget_index_voc
    parameter['input_vocab_size'] = input_vocab_size
    parameter['taget_vocab_size'] = taget_vocab_size
    parameter['input_emb'] = input_emb
    parameter['target_emb'] = target_emb
    del input_voc_index,input_index_voc,input_vocab_size,taget_voc_index,taget_index_voc,taget_vocab_size,input_emb,target_emb,bert_client,parameter_copy
    pk.dump(parameter,open('parameter.pkl','wb'))
else:
    # 读取已经处理好的parameter，但是考虑到模型训练的参数会发生变化，
    # 因此此处对于parameter中模型训练参数进行替换
    parameter = pk.load(open('parameter.pkl','rb'))
    for i in parameter_copy.keys():
        if i not in parameter:
            parameter[i] = parameter_copy[i]
            continue
        if parameter_copy[i] != parameter[i]:
            parameter[i] = parameter_copy[i]
    for i in parameter_copy.keys():
        print(i,':',parameter[i])
    pk.dump(parameter,open('parameter.pkl','wb'))
    del parameter_copy,i
    
data_set = batch_yield(parameter,np.array(ins),np.array(outs))
batch_x,batch_y,batch_y2,x_len,y_len,y2_len,epoch,keys = next(data_set)
batch_x.shape,batch_y.shape,x_len,y_len