#coding:utf-8
#整理自深兰NLP项目课(10.30):第一次构建rnn

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle as pk

def softmax(x):
    """
    softmax函数：
    此处减去最大值防止爆炸
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

#  RNN类
class RNN:
    def __init__(self, epochs=20, n_h=16, alpha=0.01, batch_size=32):
        """
        n_h -- 隐藏层维度
        n_x -- 每个cell输入xi的维度
        n_y -- 每个cell输出yi的维度
        alpha -- 梯度下降参数
        epochs -- 迭代次数
        batch_size -- 每个batch大小
        """
        self.epochs = epochs
        self.alpha = alpha
        self.parameters = {}
        self.loss = 0.0
        self.n_h = n_h #隐藏层维度需要自己设置，此处设置50
        self.n_x = 2 # 初始随便设，实际是词的维度，若是wordvect就是768维度，若是字典就是onehot的维度，本例指onehot维度，即字典字数
        self.n_y = 2
        self.m = batch_size
        
    def initialize_parameters(self, n_h, n_x, n_y):
        """
        Whx -- 为应用于输入样本的权重矩阵, 从输入到隐层, 相当于上文中的U, 形状为(n_h, n_x)
        Whh -- 为用于循环计算的权重矩阵, 相当于上文中的W, 形状为(n_h, n_h)
        Wyh -- 为应用于隐层至输出层的权重矩阵, 相当于上文中的V, 形状为(n_y, n_h)
        bh --  自输入层至隐层时的偏置, 形状为(n_h, 1)
        by --  自隐层至输出层时的偏置, 形状为(n_y, 1)
        """
        np.random.seed(1)
        Whx = np.random.randn(n_h, n_x)*0.01  # 输入至隐层 50x904
        Whh = np.random.randn(n_h, n_h)*0.01  # 上一个隐层至下一个隐层   50x50
        Wyh = np.random.randn(n_y, n_h)*0.01  # 隐层至输出 904x50
        bh = np.zeros((n_h, 1))  # 隐层的偏置
        by = np.zeros((n_y, 1))  # 输出层偏置
        self.parameters = {"Whx": Whx, "Whh": Whh, "Wyh": Wyh, "bh": bh, "by": by}
        self.n_x = n_x  #904
        self.n_y = n_y
        
    def rnn_cell_forward(self, xt, h_prev):
        """
        用于实现单个单元的前向传播，以上图为参考，实现如下：
        xt -- 为t时刻下数据的输入，形状为(n_x, m)
        h_prev -- 为t-1时刻下隐层的内容，形状为(n_h, m)
        h_next -- 为t时刻下隐层的内容，形状为(n_h, m)
        yt_pred -- 为t时刻下预测的结果，形状为(n_y, m)
        cache -- 为在反向传播时需要的内容
        """
        Whx = self.parameters["Whx"]
        Whh = self.parameters["Whh"]
        Wyh = self.parameters["Wyh"]
        bh = self.parameters["bh"]
        by = self.parameters["by"]
        # 公式可参见上图
        h_next = tanh(np.dot(Whh, h_prev) + np.dot(Whx, xt) + bh) #tanh( 50x50*50x1 + 50x904 * 904x1 + 50x1 ) ->50x1
        
        tmp = np.dot(Wyh, h_next) + by # 904x50 * 50x1 + 904x1 ->904x1
        yt_pred = softmax(tmp) #最后转换成类似[0.1,0.01,0.8,]
        # 存储反向传播所需的内容
        cache = (h_next, h_prev, xt)
        return h_next, yt_pred, cache
    
    def rnn_forward(self, x, h_prev):
        """
        x -- 为每个时刻下的输入，形状为(n_x, m, T_x)
        h_prev -- 为第0时刻下，隐层的状态，形状为(n_h, m)
        
        h -- 为每个时刻下，隐层的内容，形状为(n_h, m, T_x)
        y_pred -- 为每个时刻下的输出，形状为(n_y, m, T_x)
        caches -- 包含所有反向传播所需的内容
        """
        # 存储所有隐层的状态和输入
        caches = []
        n_x, m, T_x = x.shape #T_x 41是词的长度
        n_y, n_h = self.parameters["Wyh"].shape
        # 初始化隐藏层和输出层
        h = np.zeros((n_h, m, T_x)) #50x1x41
        y_pred = np.zeros((n_y, m, T_x))
        # 设置t0时刻，隐层的内容
        h_next = h_prev
        # 循环所有的时间状态
        for t in range(T_x):
            # 更新第t1至所有时刻的隐层和输出层的状态
            h_next, yt_pred, cache = self.rnn_cell_forward(xt=x[:, :, t], h_prev=h_next)
            h[:, :, t] = h_next
            y_pred[:, :, t] = yt_pred
            # 记忆记录
            caches.append(cache)
        # 存储反向传播所需内容
        caches = (caches, x)
        return h, y_pred, caches
    
    def rnn_cell_backward(self, dy, gradients, cache):
        """
        dy -- 为预测值与观测值得偏差
        dWhx -- 为Whx梯度，形状为(n_h, n_x)
        dWhh -- 为Whh梯度，形状为(n_h, n_h)
        dWyh -- 为Why梯度，形状为(n_y, n_h)
        dbh -- 为bh梯度，形状为(n_h, 1)
        dby -- 为by梯度，形状为(n_y, 1)
        """
        # 初始化各权重
        (h_next, h_prev, xt) = cache
        Whh = self.parameters["Whh"]
        Wyh = self.parameters["Wyh"]
        # 梯度计算
        # 根据公式dV = dy*ht的累加，偏置即批次上进行sum
        gradients['dWyh'] += np.dot(dy, h_next.T)
        gradients['dby'] += np.sum(dy, axis=1, keepdims=True)
        # 这里的梯度包含有两块，一个是时间上梯度的更新，即前层时刻传到下个cell的梯度；
        # 一个是空间上梯度的更新，来自于隐层和输出层之间
        dh = gradients['dh_next'] + np.dot(Wyh.T, dy)
        # 这里注意与Jordan矩阵相乘，相当于对应位置相乘
        dtanh = np.multiply(dh, 1 - np.square(h_next))
        # UW的更新基于dtanh
        gradients['dWhx'] += np.dot(dtanh, xt.T)
        gradients['dWhh'] += np.dot(dtanh, h_prev.T)
        gradients['dbh'] += np.sum(dtanh, axis=1, keepdims=True)
        # 更新下一次时间上的梯度
        gradients['dh_next'] = np.dot(Whh.T, dtanh)
        return gradients

    def rnn_backward(self, y, y_hat, caches):
        """
        y -- 为标签，形状为(n_y, m, T_x)
        y_hat -- 为经过softmax输出的预测值，形状为(n_y, m, T_x)
        caches -- 为用于反向传播的所需的内容
        dWhx -- 为Whx梯度，形状为(n_h, n_x)
        dWhh -- 为Whh梯度，形状为(n_h, n_h)
        dWyh -- 为Why梯度，形状为(n_y, n_h)
        dbh -- 为bh梯度，形状为(n_h, 1)
        dby -- 为by梯度，形状为(n_y, 1)
        """
        # 确定形状
        (caches, x) = caches
        n_x, m, T_x = x.shape
        # 初始化各梯度
        gradients = {}
        gradients['dWhx'] = np.zeros((self.n_h, self.n_x))
        gradients['dWhh'] = np.zeros((self.n_h, self.n_h))
        gradients['dbh'] = np.zeros((self.n_h, 1))
        gradients['dh_next'] = np.zeros((self.n_h, self.m))
        gradients['dWyh'] = np.zeros((self.n_y, self.n_h))
        gradients['dby'] = np.zeros((self.n_y, 1))
        dy = y_hat - y  
        # 遍历各个状态
        for t in reversed(range(T_x)):
            gradients = self.rnn_cell_backward(dy=dy[:, :, t], gradients=gradients, cache=caches[t])
            
        maxValue = 2
        #修饰梯度的值，在一定-maxValue至maxValue范围内，防止梯度溢出
        dWhh, dWhx, dWyh, dbh, dby = gradients['dWhh'], gradients['dWhx'], gradients['dWyh'], gradients['dbh'], gradients['dby']
        for gradient in [dWhx, dWhh, dWyh, dbh, dby]:
            np.clip(gradient, -1*maxValue, maxValue, out=gradient)
        gradients = {"dWhh": dWhh, "dWhx": dWhx, "dWyh": dWyh, "dbh": dbh, "dby": dby}
        return gradients

    def update_parameters(self, gradients):
        self.parameters['Whx'] += -self.alpha * gradients['dWhx']
        self.parameters['Whh'] += -self.alpha * gradients['dWhh']
        self.parameters['Wyh'] += -self.alpha * gradients['dWyh']
        self.parameters['bh'] += -self.alpha * gradients['dbh']
        self.parameters['by'] += -self.alpha * gradients['dby']

    def compute_loss(self, y_hat, y):
        n_y, m, T_x = y.shape
        for t in range(T_x):
            self.loss -= 1/m * np.sum(np.multiply(y[:, :, t], np.log(y_hat[:, :, t])))
        return self.loss

    def optimize(self, X, Y, h_prev):
        """
        优化器
        X -- 为输入数据序列，形状为(n_x, m, T_x)，n_x是每个step输入xi的维度，m是一个batch数据量，T_x一个序列长度
        Y -- 为每个输入xi对应的输出yi，形状为(n_y, m, T_x)，n_y是输出向量
        h_prev -- 为前一个时刻隐层的状态

        h[len(X)-1] -- 为最新时刻的隐层状态，形状为(n_h, 1)
        """
        # 正向传播
        h, y_pred, caches = self.rnn_forward(X, h_prev)
        # 计算损失
        loss = self.compute_loss(y_hat=y_pred, y=Y)
        gradients = self.rnn_backward(Y, y_pred, caches)
        self.update_parameters(gradients)
        return loss, gradients, h[:, :, -1]
    
    
def sample():
    # 初始化生成诗句词的index，值得注意此处'\n'的index为0
    idx = -1
    indices = []
    # 配置初始的输入和隐层状态
    x = np.zeros((vocab_size, 1))
    h_prev = np.zeros((n_h, 1))
    # 限制一个句子长度为40
    count = 0
    while idx != 0 and count <= 40:
        h_next, yt_pred, _ = rnn.rnn_cell_forward(x,h_prev)
        # 根据softmax概率随机挑选下一个词
        idx = np.random.choice(range(vocab_size), p=yt_pred.ravel())
        indices.append(idx)
        # 准备下一个词的向量
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        h_prev = h_next
        # 记录词的数量
        count += 1
    if indices[-1] != 0:
        indices.append(0)
    res = ''.join(ind_to_char[ix] for ix in indices)
    return res


if __name__ == "__main__":
    
    data = pd.read_csv('data/诗句.csv').content
    data_to_save = []
    with open('data/shiju.txt','w',encoding = 'utf-8') as f:
        for i in data:
            tmp = [len(k) for j in i.split('，') for k in j.split('。') if k != ''] #切分后计算各个诗句的长度
            if len(set(tmp)) < 2 and (5 == len(tmp)): #判断是否是5言律诗
                data_to_save.append(i.replace('\r','').replace('<strong>','').replace('</strong>',''))
        for i in set(data_to_save): #set为了去重
            f.write(i+'\n') #每一句后加换行符，并写入txt文件
            
    # 加载数据
    data = open('data/shiju.txt','r',encoding = 'utf-8').readlines()
    data = [i.strip() for i in data if '□' not in i] #去除头尾空格
    chars = ['\n']+list(set(''.join(data))) #join将所有诗句连接成一个字符串，set将其按字切分
    vocab_size = len(chars)#一共904个不重复的字
    char_to_ind = {ch: i for i, ch in enumerate(sorted(chars))}
    ind_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
    print(vocab_size,len(data)) #904 72
    

    
    # 配置训练参数
    n_h = 50 # 隐层维度
    #iter_num = 250000 # 迭代次数
    iter_num = 250000 # 迭代次数
    test_out_num = 5 # 训练过程中每次输出5个样例
    rnn = RNN(n_h=n_h, batch_size=1)
    n_x, n_y = vocab_size, vocab_size
    rnn.initialize_parameters(n_h, n_x, n_y)
    np.random.seed(0)
    np.random.shuffle(data)
    # 初始化t0时刻隐层状态
    h_prev = np.zeros((n_h, 1))
    
    
    min_loss = float('inf')
    for i in tqdm(range(iter_num)):
    #for i in range(iter_num):
        index = i % len(data)
        x = [None] + [char_to_ind[ch] for ch in data[index]] #data[0]:'绿杨芳草长亭路。年少抛人容易去。楼头残梦五更钟，花底离情三月雨。无情不似多情苦。'
        y = x[1:] + [char_to_ind["\n"]] #y从x的第1个元素开始，后面追加'\n'
        X_batch = np.zeros((n_x, 1, len(x))) #(904, 1, 41)
        Y_batch = np.zeros((n_y, 1, len(x))) #(904, 1, 41)
        for t in range(len(x)):
                if x[t] is not None:
                    X_batch[x[t], 0, t] = 1 #构建onehot向量
                Y_batch[y[t], 0, t] = 1   #构建onehot向量
        rnn.loss = 0
        curr_loss, gradients, h_prev = rnn.optimize(X=X_batch, Y=Y_batch, h_prev=h_prev)
        if i % 2000 == 0:
            # 模型保存
            if curr_loss < min_loss:
                min_loss = curr_loss
                pk.dump([rnn,vocab_size,char_to_ind,ind_to_char, curr_loss],open('rnn.pkl','wb'))
            print('Iteration: %d, Loss: %f' % (i, curr_loss) + '\n')
            for name in range(test_out_num):
                print(sample())
            print('\n')    