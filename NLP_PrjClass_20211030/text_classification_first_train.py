#coding:utf-8
# 来源：整理自深兰NLP项目课:第一次实现意图识别(文本分类) -->已完成
# 功能：实现了训练一个特征提取器，用于对新闻语料的分类，一共分为15个类别，训练好的模型保存为
# 备注：只实现了textRNN，是双向lstm; 有空可以实现TextCNN和TextRCNN

import torch
import os
import time
import gensim
import random
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm
from operator import itemgetter
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch.nn.functional as F   # pytorch 激活函数的类
from torch import nn,optim        # 构建模型和优化器

import os
import shutil
import pickle as pk
from torch.utils.tensorboard import SummaryWriter


# import tensorflow as tf
# torch.__version__,tf.__version__
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#torch.device('cpu')#
torch.__version__
# ,tf.__version__
# print(device,torch.__version__)
# # print(device,torch.__version__,tf.__version__)
# import tensorflow as tf
# import tensorboard
# tf.__version__,tensorboard.__version__


def build_dataSet(data,parameter):
    # # 构建训练集（chars，labels），构建词汇表（char2ind，ind2char）构建词向量ind2embeding
    #1.构建char和其次数对应的字典
    chars = [] # 382642条数据(句子)，且每条(句子)被按字切分存放
    labels = []# 数据对应的382642标签
    vocab = defaultdict(int) #保存切分后的字及对应的次数形成的字典 5548个元素，如: {'<pad>': 1, '京': 4127, '城': 6166, '最': 30089, ...}
    vocab['<pad>'] = parameter['min_count_word']
    
    for text,label in tqdm(zip(data.text,data.label)):
        chars.append(text.split()) #按字切分后存入chars中
        labels.append(label)
        for char in chars[-1]:# append是在list尾部插入，chars[-1]就代表刚刚插入的那条text
            vocab[char] += 1 #统计字出现的字数,共5548个不重复的char
    vocab['<unk>'] = parameter['min_count_word']
    
    for char in vocab: #取得是vocab.keys()
        if vocab[char] < parameter['min_count_word']: #vocab[char] 拿到的是vocab.values()中的每一个数字
            del vocab[char] #次数< min_count_word 的字直接删掉
    print(len(vocab))    #5548
    
    #构造char->index和index->char对应的字典
    char2ind,ind2char = dict(zip(vocab.keys(),range(len(vocab)))), dict(zip(range(len(vocab)),vocab.keys()))
    ind2embeding = np.random.randn(len(vocab), parameter['embedding_dim']).astype(np.float32) / np.sqrt(len(vocab)) #随机初始化5548x300
    
    # 2. 加载词向量，确定每个char对应的300维的词向量
    w2v = gensim.models.Word2Vec.load('data/wiki.Mode')
    
    for ind,i in enumerate(char2ind.keys()): #ind是索引，i才是取到的char,char2ind:['<pad>', '京', '城', '最', '值', '得', '你', '来', '场', ...,]
        try:
            embedding = np.asarray(w2v.wv[i], dtype='float32') #从w2v中取char对应的300维的wordvect
            ind2embeding[ind] = embedding #ind 对应的i是一个字，这里按顺序拿到每个字在wiki.Mode中对应的词向量
        except:
            parameter['num_unknow'] += 1 #wiki.Mode中没有的char
    #这里有个疑问，parameter['num_unknow']=253,但是ind2embeding却有5548个元素，没有抛弃的char吗? char2ind.keys()也是5548个元素。
      #解答:因为ind2embeding被初始化为5548x300,若有异常抛出，即wiki.Mode中未查到该char的vector，但就用初始化的vector
      
    parameter['ind2char'] = ind2char  #index->字  这里index跟频次无关，按顺利来
    parameter['char2ind'] = char2ind  #字->index
    parameter['ind2embeding'] = ind2embeding #5548x300
    parameter['output_size'] = len(set(labels))
    return np.array(chars),np.array(labels)

def batch_yield(chars,labels,parameter,shuffle = True):#chars.__len__():306113
    for train_epoch in range(parameter['epoch']):
        if shuffle:
            permutation = np.random.permutation(len(chars))
            chars = chars[permutation]    #打乱顺序
            labels = labels[permutation]  #打乱顺序
        max_len = 0
        batch_x,batch_y = [],[] #是一个batch的训练集
        #循环306113,每次满足len(batch_x) = parameter['batch_size']生成一个batch的训练集
        for iters in tqdm(range(len(chars))): 
            
            #从parameter['char2ind']中取chars[iters]对应一个句子的每个char的index
            batch_ids = itemgetter(*chars[iters])(parameter['char2ind'])
            try:
                batch_ids = list(batch_ids) #元祖转换成list，一个元素时转换失败
            except:
                batch_ids = [batch_ids,0]
            if len(batch_ids) > max_len:
                max_len = len(batch_ids)
            batch_x.append(batch_ids)
            batch_y.append(labels[iters])
            #print(f"max_len:{max_len}")
            
            #其实是当len(batch_x) = parameter['batch_size']的时候再将index向量化
            if len(batch_x) >= parameter['batch_size']:
                
                #将由index构成的batch_x转换成由wordvect构成;
                #根据x_ids查wordvect，并用<pad>对应的embeding补齐到max_len，parameter['ind2embeding'][0]是<pad>对应的embeding
                batch_x = [np.array(list(itemgetter(*x_ids)(parameter['ind2embeding']))+[parameter['ind2embeding'][0]]*(max_len-len(x_ids))) for x_ids in batch_x] 
                device = parameter['cuda']
                
                #向量化后batch_x:10x31x300，其中31是这个batch的最大长度，每个batch不一样，这在后面训练不会有问题吗？
                yield torch.from_numpy(np.array(batch_x)).to(device),torch.from_numpy(np.array(batch_y)).to(device).long(),True,None
                max_len,batch_x,batch_y = 0,[],[]
                
        #最后剩余少于batch_size再用下面代码做处理
        batch_x = [np.array(list(itemgetter(*x_ids)(parameter['ind2embeding']))+[parameter['ind2embeding'][0]]*(max_len-len(x_ids))) for x_ids in batch_x]
        device = parameter['cuda']
        
        yield torch.from_numpy(np.array(batch_x)).to(device),torch.from_numpy(np.array(batch_y)).to(device).long(),True,train_epoch
        max_len,batch_x,batch_y = 0,[],[]
        
    yield None,None,False,None


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


if __name__== "__main__":
    
    # 准备好模型的参数;这是全局变量
    parameter = {   
        'min_count_word':1,
        'char2ind':None,
        'ind2char':None,
        'ind2embeding':None,
        'output_size': None, #由分类的标签数决定
        'epoch':20,
        'batch_size':10,
        'embedding_dim':300,
        'hidden_size':128,
        'num_layers':2, #堆叠LSTM的层数，默认值为1
        'dropout':0.5,
        'cuda':device,
        'lr':0.001,
        'num_unknow':0
    }
    

#准备数据====================================================================================
    if os.path.exists('dataSet_textClssi.pkl') and os.path.exists('parameter_textClssi.pkl'): #如果存在直接open
        
        [train_chars,test_chars,train_labels,test_labels] = pk.load(open('dataSet_textClssi.pkl','rb'))
        
        parameter = pk.load(open('parameter_textClssi.pkl','rb'))
        device = torch.device('cpu')
        parameter['cuda'] = device
    else: #不存在则创建
        data = pd.read_csv('data/classification_data.csv')
        print(data[0:10]) #查看下数据格式,主用的inidex列是lable和text
        
        #得到的chars元素是382642个切分了的句子，lables是对应句子的382642个标签
        #该接口还生成了:ind2char  #index->字  这里index跟频次无关，按顺利来,# char2ind ,字->index #ind2embeding ,5548x300的词向量
        chars_src,labels_src = build_dataSet(data,parameter)
        
        # 按比例划分训练集和测试集
        train_chars,test_chars, train_labels,test_labels = train_test_split(chars_src,labels_src, test_size=0.2, random_state=42)
        pk.dump([train_chars,test_chars,train_labels,test_labels],open('dataSet_textClssi.pkl','wb'))
        pk.dump(parameter,open('parameter_textClssi.pkl','wb'))
    
    #构建迭代器------------------------------------
    train_yield = batch_yield(train_chars,train_labels,parameter)
    
    #seqs::10x24x300 label::10x1  keys::True  其中24为一个batch中所有句子最大长度，每个batch可能不一样,每个batch长度不一致，后面训练不会有问题吗？
    seqs,label,keys,epoch = next(train_yield)
    
    #sta = time.time()
    #while 1:
        #a,b,c,d = next(train_yield)
        #if not c:
            #break
            
    # print(time.time()-sta)
    # train_chars
    seqs,labels,_,_ = next(train_yield)
    print('\n',seqs[:10].shape,labels[:10])
    
    
#模型训练==============================================================================

    # 记录日志
    shutil.rmtree('textrnn') if os.path.exists('textrnn') else 1
    writer = SummaryWriter('./textrnn', comment='textrnn') #报错暂时无法解决
    
    # 构建模型
    model = TextRNN(parameter).to(parameter['cuda'])
    
    # 确定训练模式
    model.train()
    
    # 确定优化器和损失
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.95, nesterov=True)
    # optimizer = torch.optim.Adam(model.parameters(),lr = parameter['lr'], \
    #                              weight_decay = 0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 保存图
    train_yield = batch_yield(train_chars,train_labels,parameter)
    seqs,label,keys,epoch = next(train_yield)
    writer.add_graph(model, (seqs,))
    
    # 准备迭代器
    train_yield = batch_yield(train_chars,train_labels,parameter)
    
    # 开始训练-------------------------------------------------------
    loss_cal = []
    min_loss = float('inf')
    #count=0
    with writer:
        while 1:
        #while count <10:
            #count=count+1
            seqs,label,keys,epoch = next(train_yield)
            if not keys:
                break
            
            out = model(seqs) #输出是10x15 10个样本句子，每个样本都有15个标签的概率
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_cal.append(loss.item())
            if epoch is not None:
                if (epoch+1)%1 == 0:
                    loss_cal = sum(loss_cal)/len(loss_cal)
                    if loss_cal < min_loss:
                        min_loss = loss_cal
                        torch.save(model.state_dict(), 'model-rnn1.h5')
                    print('epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, parameter['epoch'],loss_cal))
                    
                writer.add_scalar('loss',loss_cal,global_step=epoch+1)
                loss_cal = [loss.item()]
        writer.flush()
        writer.close()    