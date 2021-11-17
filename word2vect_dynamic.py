#coding:utf-8
import jieba
import pandas as pd
from tqdm import tqdm
# 获取原始数据
data_math = pd.read_csv('../data/数学原始数据.csv',encoding = 'gbk',header = None)[0]
data = list(data_math)
del data_math

# 因为bert所设计中，其中一块任务是判断下一句内容是否相关
# 采取的方法是，切分逗号和句号；逗号句号，前后句子相关
data4bert = []
for i in data:
    # 切分句子
    for j in i.split('，'):
        for k in j.split('。'):
            if k != '':
                data4bert.append(k)
    # 添加一个标志信息，判断相关的句子结束
    data4bert.append('\n')

stop_words = '。，、（）().,:：\\、"“”；;||？?^<>シ[]·【】け√《》°{}\u3000'#abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890+-/
data_after_cut = []
# 对处理的数据进行清洗和切词
for i in tqdm(data4bert):
    cut_sen = [j for j in jieba.lcut(i) if j not in stop_words]
    if len(cut_sen) >= 2 or cut_sen == ['\n']:
        data_after_cut.append(cut_sen)
        
data_after_cut[:10]


from collections import defaultdict
from operator import itemgetter
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import random
import torch
import pickle as pk

def build_vocab(corpus,parameter):
    vocab = defaultdict(int)
    # 此处是bert训练时候设置的标志位，<PAD>用于补齐,<SOS>确定起始位，<EOS>确定结束位，
    # <MSK>此处是bert训练的另一个任务，预测掩码，<UNK>用于未知词的标记
    vocab['<PAD>'] = parameter['min_freq']
    vocab['<SOS>'] = parameter['min_freq']
    vocab['<EOS>'] = parameter['min_freq']
    vocab['<MSK>'] = parameter['min_freq']
    vocab['<UNK>'] = parameter['min_freq']
    # 计算各词出现的频率
    for i in corpus:
        for j in i:
            vocab[j] += 1
    # 清除低频词
    for i in vocab:
        if vocab[i] < parameter['min_freq']:
            del vocab[i]
    # 确定词表大小
    vocab_size = len(vocab) #5239
    # 构建词-》id，id-》词的字典
    voc_index = dict(zip(vocab.keys(),range(len(vocab.keys()))))
    index_voc = dict(zip(range(len(vocab.keys())),vocab.keys()))
    parameter['voc_index'] = voc_index
    parameter['index_voc'] = index_voc
    parameter['vocab_size'] = vocab_size
    corpus_new =[ ]#只保存成对的句子
    # 提前准备好数据集，两者是相关的数组；如[['我今天去了人民广场','很开心']]；准备后续基于上述规则进行调整
    for ind,i in enumerate(corpus):
        if ind+1 < len(corpus):
            l = i
            r = corpus[ind+1]
            # 若出现\n表明两者句子并不相关，因此进行跳过
            if l == ['\n'] or r == ['\n']:
                continue
            corpus_new.append([l,r])
    return corpus_new
    
def batch_yield(corpus,parameter,shuffle = True):
    for epoch in range(parameter['epoch']):
        if shuffle:
            random.shuffle(corpus) #把句子打乱，但还是成对
        ins_list,label_list,seg_list,next_label_list = [],[],[],[]
        for iters in tqdm(range(len(corpus))):
            # 其中iters类似于['我今天去了人民广场','很开心']
            # 那么基于规则2.1和2.2,有如下处理
            t1,t2,next_label = None,None,None
            if random.random() > 0.5:
                # 两者相关，标签为1
                t1,t2 = corpus[iters]
                next_label = 1
            else:
                # 两者不相关，标签为0
                t1 = corpus[iters][0]
                t2 = corpus[sample(iters,len(corpus))][1]
                next_label = 0
            # 基于规则1.1、1.2、1.3，构建掩码预测的数据
            t1_random, t1_label = random_word(t1,parameter) #将句子mask
            t2_random, t2_label = random_word(t2,parameter)
            # 前一个句子和后一个句子分别填充标志位
            t1 = [parameter['voc_index']['<SOS>']]+t1_random+[parameter['voc_index']['<EOS>']]
            t2 = t2_random+[parameter['voc_index']['<EOS>']]
            # 对应待预测的标签同样添加标志位
            t1_label = [parameter['voc_index']['<PAD>']] + t1_label + [parameter['voc_index']['<PAD>']]
            t2_label = t2_label + [parameter['voc_index']['<PAD>']] 
            # 构建区分前一个句子和下一个句子区分的信息，主要用于SegmentEmbedding
            segment_label = ([1]*len(t1)+[2]*len(t2))[:parameter['max_len']]
            # 生成模型输入和用于掩码预测的标签
            ins = (t1+t2)[:parameter['max_len']]
            label = (t1_label+t2_label)[:parameter['max_len']]
            if len(segment_label) < parameter['max_len']:
                pad = [parameter['voc_index']['<PAD>']]*(parameter['max_len']-len(segment_label))
                ins.extend(pad), label.extend(pad), segment_label.extend(pad)
            ins_list.append(ins)
            label_list.append(label)
            seg_list.append(segment_label)
            # 下一个句子相关性预测标签，相关1，不相关0
            next_label_list.append(next_label)
            # 当list大小等于batch_size，数据输出
            # 输出内容包括有模型输入t1+t2，掩码预测标签（结果），用于SegmentEmbedding的信息，和用于下一个句子相似的标签
            if len(ins_list) >= parameter['batch_size']:
                yield list2torch(ins_list),list2torch(label_list),list2torch(seg_list),list2torch(next_label_list),None,False
                ins_list,label_list,seg_list,next_label_list = [],[],[],[]
        yield list2torch(ins_list),list2torch(label_list),list2torch(seg_list),list2torch(next_label_list),epoch,False
    yield None,None,None,None,None,True
            
def list2torch(a):
    return torch.from_numpy(np.array(a)).long()
            
def random_word(ins,parameter):
    new_ins = deepcopy(ins) #array
    out_label = []
    for i,word in enumerate(new_ins):
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15
            # 其中的80%使用<MSK>进行替换，
            if prob < 0.8:
                new_ins[i] = parameter['voc_index']['<MSK>']
            # 其中的10%使用随机的id进行替换
            elif prob < 0.9:
                new_ins[i] = random.randrange(parameter['vocab_size'])
            # 其中的10%是原始的id不过需要进行预测
            else:
                new_ins[i] = parameter['voc_index'][word]
            out_label.append(parameter['voc_index'][word])
        else:
            new_ins[i] = parameter['voc_index'][word]
            out_label.append(0)
    return new_ins,out_label #new_ins:['这', '说明', '当', '指数', '为', '零时'] new_ins:[34, 35, 3, 37, 38, 39] out_label:[0, 0, 36, 0, 0, 0] 其中'<MSK>': 3,
    
            
def sample(ind,max_size):
    while 1:
        ind_sample = random.randrange(max_size)
        if ind != ind_sample:
            return ind_sample
        
parameter = {
    'min_freq':1,
    'epoch':1000,
    'batch_size':50,
    'd_model':64,
    'dropout':0.,
    'd_q':8,
    'd_k':8,
    'd_v':8,
    'n_heads':8,
    'd_ff':2048,
    'n_layers':8,
    'device':torch.device('cuda'),
    'max_len':40,
}
device = parameter['device']
corpus = build_vocab(data_after_cut,parameter) #返回句子对
print(corpus[1][0])
print([parameter['voc_index'][i] for i in corpus[1][0]])
pk.dump(parameter,open('parameter.pkl','wb'))
random_word(corpus[1][0],parameter),len(corpus)

train_yield = batch_yield(corpus,parameter,32)
ins,label,segment_label,next_label,epoch,keys = next(train_yield)