#coding:utf-8
#第一次用bert构建动态词向量
#第一步：获取数据并清洗
import jieba
import pandas as pd
from tqdm import tqdm
# 获取原始数据
data_math = pd.read_csv('../data/数学原始数据.csv',encoding = 'gbk',header = None)[0] #5725 sentences作为一个list，每个元素是一个句子
data = list(data_math)
del data_math

# 因为bert所设计中，其中一块任务是判断下一句内容是否相关
# 采取的方法是，切分逗号和句号；逗号句号，前后句子相关
data4bert = []#存放按逗号分隔后的句子片段，每个句号结尾的句子添加标志 '\n'
for i in data:
    # 切分句子
    for j in i.split('，'):#每个句子用逗号划分成不同部分
        for k in j.split('。'):#逗号前后的部分再用句号划分实际是去掉句号，因为句号后有个空格
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

#一句话切分的结果:
#[['掌握', '用', '零点', '分段法', '解高次', '不等式', '和', '分式', '不等式'], ['特别', '要', '注意', '因式', '的', '处理', '方法'], ['\n']]
data_after_cut[:10]


#第二步：调整数据
from collections import defaultdict
from operator import itemgetter
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import random
import torch
import pickle as pk

def build_vocab(corpus,parameter):
    vocab = defaultdict(int)#字典
    # 此处是bert训练时候设置的标志位，<PAD>用于补齐,<SOS>确定起始位，<EOS>确定结束位，
    # <MSK>此处是bert训练的另一个任务，预测掩码，<UNK>用于未知词的标记
    vocab['<PAD>'] = parameter['min_freq']
    vocab['<SOS>'] = parameter['min_freq']
    vocab['<EOS>'] = parameter['min_freq']
    vocab['<MSK>'] = parameter['min_freq']
    vocab['<UNK>'] = parameter['min_freq']
    # 计算各词出现的频次
    for i in corpus:
        for j in i:
            vocab[j] += 1
    # 清除低频词
    for i in vocab:
        if vocab[i] < parameter['min_freq']:
            del vocab[i]
    # 确定词表大小
    vocab_size = len(vocab)
    # 构建词-》id，id-》词的字典
    voc_index = dict(zip(vocab.keys(),range(len(vocab.keys()))))
    index_voc = dict(zip(range(len(vocab.keys())),vocab.keys()))
    parameter['voc_index'] = voc_index
    parameter['index_voc'] = index_voc
    parameter['vocab_size'] = vocab_size
    corpus_new =[ ]
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
            random.shuffle(corpus)
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
            t1_random, t1_label = random_word(t1,parameter)
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
            
def random_word(ins,parameter): #传入的是分隔后的句子
    new_ins = deepcopy(ins)
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
            new_ins[i] = parameter['voc_index'][word] #new_ins本来是句子，逐步被替换成对应number，替换的过程进行mask操作
            out_label.append(0)
    return new_ins,out_label  #原corpus[i][j]对应的number，同时进行了mask等操作，out_label保存了mask前的词元对应的number，0代表没进行mask
    
            
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
    #'device':torch.device('cuda'),
    'device':torch.device('cpu'),
    'max_len':40,
}
device = parameter['device']

corpus = build_vocab(data_after_cut,parameter) 

# 得到句子对 如 corpus[1]:[['这', '说明', '当', '指数', '为', '零时'], ['幂', '的', '值', '是', '有', '意义', '的']]
print(corpus[1][1]) # ['幂', '的', '值', '是', '有', '意义', '的']
print([parameter['voc_index'][i] for i in corpus[1][1]]) # [40, 17, 41, 42, 43, 44, 17]
pk.dump(parameter,open('parameter.pkl','wb'))

# ([40, 17, 41, 42, 43, 44, 3], [0, 0, 0, 0, 0, 44, 17]) 8395
print(random_word(corpus[1][1],parameter),len(corpus) )#

train_yield = batch_yield(corpus,parameter,32)
ins,label,segment_label,next_label,epoch,keys = next(train_yield)

#第三步bert构建
import torch.utils.data as Data
from tqdm import tqdm
import torch.nn as nn
import pickle as pk
import numpy as np
import torch
import math
import os

# 位置层编码，和原始的transformer一致
# BERT学习到输入的顺序属性
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len = 512):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model) #512x64
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #512x1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #32x1
        pe[:, 0::2] = torch.sin(position * div_term) #512x32 [:, 0::2]是从0开始，每隔一列取一列
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:, :x.size(1)]

# 用于BERT去区分一个句子对中的两个句子，前一个句子输入的是全为1的id，后一个句子输入的是全为2的id这样进行区分，
# 如果只有一个句子那么就是都为0
# 辅助BERT区别句子对中的两个句子的向量表示
class SegmentEmbedding(nn.Embedding):
    def __init__(self, d_model):
        super().__init__(3, d_model, padding_idx=0)
        
# 对词进行嵌入，编码；我们经常为了快速收敛直接使用已经训练好的词向量，此处是直接随机化一个可训练的向量表征这个词
# 词的向量表示
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super().__init__(vocab_size, d_model, padding_idx=0)
        
class bertEmbedding(nn.Module):
    def __init__(self,parameter):
        super().__init__()
        vocab_size = parameter['vocab_size']
        d_model = parameter['d_model'] #64
        dropout = parameter['dropout'] # 0
        
        self.token = TokenEmbedding(vocab_size = vocab_size , d_model=d_model)
        self.position = PositionalEmbedding(d_model = d_model)
        self.segment = SegmentEmbedding(d_model = d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self,seqs,seg_label):
        x = self.token(seqs) + self.position(seqs) + self.segment(seg_label)
        x = self.dropout(x)
        return x
    
# 此处的transformer没有什么差别
class ScaledDotProductAttention(nn.Module):
    def __init__(self,parameter):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = parameter['d_k']

    def forward(self, Q, K, V, attn_mask = None):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        # 基于上述公式进行点乘相似度
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) 
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9) 
        # 经过softmax输出的是注意力或者说打分结果
        attn = nn.Softmax(dim=-1)(scores)
        # 与类似于原始输入相乘，来强化或者减弱其中的特征
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,parameter):
        super(MultiHeadAttention, self).__init__()
        device = parameter['device']
        self.d_q,self.d_k,self.d_v,self.d_model,self.n_heads = parameter['d_q'],parameter['d_k'], \
        parameter['d_v'],parameter['d_model'],parameter['n_heads']
        self.W_Q = nn.Linear(self.d_model, self.d_q * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)
        self.sdp = ScaledDotProductAttention(parameter).to(device)
        self.add_norm = nn.LayerNorm(self.d_model)
        
    def forward(self, input_Q, input_K, input_V, attn_mask = None):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        # 所谓的多头自注意机制，就是输入的内容都一致，然后通过三个不同的线性变换得到我们的QKV，然后基于点乘相似度（点乘得分函数）
        # 进行计算得到相应相应的注意力来强化或者减弱不同头的特征
        residual, batch_size = input_Q, input_Q.size(0)
        # 原始输入经过不同线性变化得到QKV
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_q).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_q]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # 基于点乘相似度得到经过注意力加成后的结果（经过强化或者减弱其中特征）
        context, attn = self.sdp(Q, K, V, attn_mask = attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, len_q, n_heads * d_v]
        # 交给全连接进行线性变化
        output = self.fc(context) # [batch_size, len_q, d_model]
        # add+ln
        output = self.add_norm(output + residual)
        return output
    
# bert的ffn和原始transformer的主要区别在于激活函数从原有的Relu变为GELU
class GELU(nn.Module):
    # bert原文中使用的是GELU
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,parameter):
        self.d_ff,self.d_model = parameter['d_ff'],parameter['d_model']
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            GELU(),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )
        self.add_norm = nn.LayerNorm(self.d_model)
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        # 所谓的ffn就是两次线性变化，中间一个激活函数，提供的就是非线性结果，加强模型表现，也可以去除；
        # 值得注意的是bert里面ffn使用的激活函数是gelu
        output = self.fc(inputs)
        return self.add_norm(output + residual) # [batch_size, seq_len, d_model]
    
# 此处与原始的transformerEncoder一致
class TransformerBlock(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.dropout = parameter['dropout']
        self.mal_attn = MultiHeadAttention(parameter)
        self.pos_ffn = PoswiseFeedForwardNet(parameter)
        self.dropout = nn.Dropout(p=self.dropout)
    
    def forward(self, x, mask):
        x = self.mal_attn(x, x, x, mask)
        output = self.pos_ffn(x)
        return self.dropout(x)
    
# 在多个transformerEncoder上补上原始词embedding和位置embedding以及用于区分两个句子的SegmentEmbedding，其他与transformer一致
class BERT(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        n_layers = parameter['n_layers'] # 8
        self.embedding = bertEmbedding(parameter)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(parameter) for _ in range(n_layers)])
        
    def forward(self, x, segment_info):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x, segment_info)
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        return x
    
model = BERT(parameter)
print(ins.shape,segment_label.shape )# 50x40
print(model(ins,segment_label).shape)  # 50x40x64

# bert特有的内容，同样是各位同学后续的主要任务，微调bert模型，扩展bert任务
# 很多大厂已经发布基于大语料训练的大模型，其他厂除非特殊情况基本不会再对特征特区模块进行再训练
# 一般就是训练以下内容，或者扩充以下内容，用于各项任务的延伸-即所谓的微调
class Bert4W2V(nn.Module):
    def __init__(self,bert, parameter):
        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(parameter)
        self.mask = MaskModel(parameter)

    def forward(self, x, segment_label):
        # bert基于大语料进行训练，得到很强的特征，所谓的强特征主要有以下几个原因：
            # 1、其中包括有（词信息、句子信息、位置信息等）
            # 2、所包含的自注意力机制，已经对于信息经过不断的迭代，强化或者弱化最后得到很强的特征
        # 一般针对于这些内容不需要进行再训练
        x = self.bert(x, segment_label)
        # 将bert捕获到的特征交给各项任务进行输出，原始bert就是两个任务，下一个句子判断和掩码预测
        return self.next_sentence(x),self.mask(x)

# 判断是否是下一个句子的任务
class NextSentencePrediction(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        hidden = parameter['d_model']
        # 经过bert获取的特征，只需经过线性变换进行输出，当然也可以改变输出网络
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))

# 掩码预测的任务
class MaskModel(nn.Module):
    def __init__(self, parameter):
        super().__init__()
        vocab_size = parameter['vocab_size']
        hidden = parameter['d_model']
        # 经过bert获取的特征，只需经过线性变换进行输出，当然也可以改变输出网络
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x)) 
    
bert = BERT(parameter).double().to(device)
bert4w2v = Bert4W2V(bert,parameter).double().to(device)
a,b = bert4w2v(ins.long().to(device),segment_label.long().to(device))    

#第四步 训练模型
import os
import shutil
import torch.optim as optim
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

# 记录日志
writer = SummaryWriter('./bert4w2v', comment='bert4w2v')

# 构建模型
bert = BERT(parameter).double().to(device)
model = Bert4W2V(bert,parameter).double().to(device)

# 确定训练模式
model.train()

# 确定优化器和损失
optimizer = Adam(model.parameters(), lr=10**-3, betas=(0.9, 0.999))
# optimizer = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.9, nesterov=True,weight_decay=0.001)
# 准备学习率策略
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
criterion = nn.NLLLoss()

print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))

# 准备迭代器
train_yield = batch_yield(corpus,parameter)

# 记录loss
avg_loss = 0.0
total_correct = 0
total_element = 0
min_loss = float('inf')
data_iter = 0
mask_loss_avg = 0.0
pred_loss_avg = 0.0
while 1:
    # 获取相关数据，模型输入，两个句子的标志位，2个标签（面向于两个任务）
    ins,label,segment_label,next_label,epoch,keys = next(train_yield)
    if not keys:
        ins = ins.long().to(device)
        label = label.long().to(device)
        segment_label = segment_label.long().to(device)
        next_label = next_label.long().to(device)
        next_sent_output, mask_lm_output = model(ins,segment_label)
        # 计算两个任务的两个损失
        next_loss = criterion(next_sent_output, next_label)
        mask_loss = criterion(mask_lm_output.transpose(1, 2), label)
        # 两个任务重要性一致，损失的重要性一致
        loss = next_loss+mask_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 计算句子相似的准确率
        correct = next_sent_output.argmax(dim=-1).eq(next_label).sum().item()
        # 统计各项损失
        mask_loss_avg += mask_loss.item()
        pred_loss_avg += next_loss.item()
        avg_loss += loss.item()
        total_correct += correct
        total_element += next_label.nelement()
        data_iter += 1
        # 每训练5次，结果进行一次输出，以防训练出现异常
        if data_iter%5 == 0:
            print('Loss: {:.4f}，avg_acc：{:.4f}，mask_loss：{:.4f}，pred_loss：{:.4f}'.format(\
                        avg_loss/data_iter,total_correct * 100.0 / total_element,mask_loss_avg/data_iter,\
                 pred_loss_avg/data_iter))
        
        if epoch is not None:
            if (epoch+1)%1 == 0:
                loss_cal = avg_loss/data_iter
                if loss_cal < min_loss:
                    # 若损失较低，模型保存
                    min_loss = loss_cal
                    torch.save(model.state_dict(), 'model-bert4w2v.h5')
                for param_group in optimizer.param_groups:
                    now_lr = param_group['lr']
                print('epoch [{}/{}], Loss: {:.4f}，avg_acc：{:.4f}，mask_loss：{:.4f}，pred_loss：{:.4f}，lr:{:.4f}'.format(epoch+1, \
                                                       parameter['epoch'],loss_cal,total_correct * 100.0 / total_element,mask_loss_avg/data_iter,\
                 pred_loss_avg/data_iter,now_lr))
                scheduler.step()
        
    else:
        break