
"""
orginal from ：
https://github.com/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer.ipynb


"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math


def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)



# # 10
def get_attn_subsequent_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


# # 7. ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
	
        # 输入进来的维度分别是 [batch_size x n_heads x len_q x d_k]  K： [batch_size x n_heads x len_k x d_k]  V: [batch_size x n_heads x len_k x d_v] ::1x8x5x64
        # 首先经过matmul函数得到的scores形状是 : [batch_size x n_heads x len_q x len_k] ::1x8x5x5
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) #Q::1x8x5x64 dot K.transpose(-1, -2)::1x8x64x5 ->1x8x5x5

        # 然后关键词地方来了，下面这个就是用到了我们之前重点讲的attn_mask，把被mask的地方置为无限小，softmax之后基本就是0，对q的单词不起作用
        # Fills elements of self tensor with value where mask is one. 1x8x5x5最后一列填充成极小值，因为其对应的是PAD;注意encoder和decoder的mask不同，前面说的只是encoder的，decodermask是上三角矩阵
        scores.masked_fill_(attn_mask, -1e9) 
	
	#对于encoder,1x8x5x5的每一行做softmax,最后一列由于补了极小值，所以softmax后概率约等于0；
        attn = nn.Softmax(dim=-1)(scores) 
	
        context = torch.matmul(attn, V) #1x8x5x5 dot 1x8x5x64 ->1x8x5x64 attn是相似度
        return context, attn


# # 6. MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
	
        # 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(d_model, d_k * n_heads) #512x64*8->512x512
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
	
	#Add&Norm
        self.linear = nn.Linear(n_heads * d_v, d_model)#8*64x512->512x512
        self.layer_norm = nn.LayerNorm(d_model) #归一化

    def forward(self, Q, K, V, attn_mask):

        # 这个多头分为这几个步骤，首先映射分头，然后计算atten_scores，然后计算atten_value;
        #输入进来的数据形状： Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model] ::1x5x512
        residual, batch_size = Q, Q.size(0) # size() 分别取1x5x512中的一个维度数，比如size(1)就是5
	
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
	#  1x5x512       -> 1x5x512         -> 1x5x8x64           ->  1x8x5x64
	
        #下面这个就是先映射，后分头；一定要注意的是q和k分头之后维度是一致额，所以一看这里都是d_k,这里但看每一个头就是讲embedding后的5x512数据，经过线性变换映射成5x512在分头成5x8x64
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k] ::QxW_Q::1x5x512 x 512x512::1x5x512::1x5x8x64(view)::1x8x5x64(transpose)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        # 输入进行的attn_mask形状是 batch_size x len_q x len_k，然后经过下面这个代码得到 新的attn_mask : [batch_size x n_heads x len_q x len_k]，就是把pad信息重复了n个头上
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) #1x5x5->1x8x5x5

        # 然后我们计算 ScaledDotProductAttention 这个函数，去7.看一下
        # 得到的结果有两个：context: [batch_size x n_heads x len_q x d_v]::1x8x5x64, attn: [batch_size x n_heads x len_q x len_k]::1x8x5x5最后一列是0
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)#这里是难点!! 得到原始x::enc_input (1x5x512) 分头后 1x8x5x64,再经过自注意力运算后的Z0-Z8::1x8x5x64
	
	#context: [batch_size x len_q x n_heads * d_v] ::1x5x512
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # Z0-Z8,8个头合在一起变成Z::1x5x512
	
        output = self.linear(context) #1x5x512 dot 512x512 ->1x5x512
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model] ::1x5x512


# # 8. PoswiseFeedForwardNet
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1) #卷积不是越卷越小吗，怎么可能从512卷积成2048？推测是每个kernel卷积出1个特征，一共2048个卷积核
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]::1x5x512
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2))) #output::1x2048x5 为什么是这样？
        output = self.conv2(output).transpose(1, 2) #重新由1x2048X5 变成1x5x512
        return self.layer_norm(output + residual) #残差



# # 4. get_attn_pad_mask

## 比如说，我现在的句子长度是5，在后面注意力机制的部分，我们在计算出来QK转置除以根号之后，softmax之前，我们得到的形状
## len_input * len*input  代表每个单词对其余包含自己的单词的影响力

## 所以这里我需要有一个同等大小形状的矩阵，告诉我哪个位置是PAD部分，之后在计算计算softmax之前会把这里置为无穷大；

## 一定需要注意的是这里得到的矩阵形状是batch_size x len_q x len_k，我们是对k中的pad符号进行标识，并没有对k中的做标识，因为没必要

## seq_q 和 seq_k 不一定一致，在交互注意力，q来自解码端，k来自编码端，所以告诉模型编码这边pad符号信息就可以，解码端的pad信息在交互注意力层是没有用到的；

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking ::[[1, 2, 3, 4, 0]]->[[[False, False, False, False,  True]]]
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k ::[[[False, False, False, False,  True]]]->变成5x5


# # 3. PositionalEncoding 代码实现
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        ## 位置编码的实现其实很简单，直接对照着公式去敲代码就可以，下面这个代码只是其中一种实现方式；
        ## 从理解来讲，需要注意的就是偶数和奇数在公式上有一个共同部分，我们使用log函数把次方拿下来，方便计算；
        ## pos代表的是单词在句子中的索引，这点需要注意；比如max_len是128个，那么索引就是从0，1，2，...,127
        ##假设我的demodel是512，2i那个符号中i从0取到了255，那么2i对应取值就是0,2,4...510
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) #max_len是什么？
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)## 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)##这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，补长为2，其实代表的就是奇数位置
        ## 上面代码获取之后得到的pe:[max_len*d_model]

        ## 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)  ## 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# # 5. EncoderLayer ：包含两个部分，多头注意力机制和前馈神经网络
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention() #多头注意力，不改变数据形状
        self.pos_ffn = PoswiseFeedForwardNet()  #前馈，其中包含残差相加，也不改变数据形状，故最终EncodeLayer输入输出数据形状一致

    def forward(self, enc_inputs, enc_self_attn_mask):
        #下面这个就是做自注意力层，输入是enc_inputs，形状是[batch_size x seq_len_q x d_model] 需要注意的是最初始的QKV矩阵是等同于这个输入的，去看一下enc_self_attn函数 6.
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # 传入enc_inputs to same Q,K,V 传出enc_outputs 1x5x512
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


# # 2. Encoder 部分包含三个部分：词向量embedding，位置编码部分，注意力层及后续的前馈神经网络

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)  ## 这个其实就是去定义生成一个矩阵，大小是 src_vocab_size * d_model
        self.pos_emb = PositionalEncoding(d_model) ## 位置编码情况，这里是固定的正余弦函数，也可以使用类似词向量的nn.Embedding获得一个可以更新学习的位置编码
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]) ## 使用ModuleList对多个EncoderLayer进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来；

    def forward(self, enc_inputs):
        # # 这里我们的 enc_inputs 形状是： [batch_size x source_len]

        # # 下面这个代码通过src_emb，进行索引定位，enc_outputs输出形状是[batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs) #[[1, 2, 3, 4, 0]] 1x5-->embedding后变成::1x5x512 每一行代表一个word的vect

        # # 这里就是位置编码，把两者相加放入到了这个函数里面，从这里可以去看一下位置编码函数的实现；3.
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) #在原有1x5x512的word vect上叠加上位置信息，最后还是1x5x512

        # #get_attn_pad_mask是为了得到句子中pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响，去看一下这个函数 4.
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) #1x5x5最后一列是True，其他元素全是False，为什么是最后一列，不是最后一行
        enc_self_attns = []
        for layer in self.layers:
            # 去看EncoderLayer 层函数 5. 
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask) #传入 1x5x512  1x5x5 返回 1x5x512 1x8x5x5
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns #enc_outputs是最终Ecoder模块的输出1x5x5132 enc_self_attns保存了6层Encoder的enc_self_attn(1x8x5x5)

# # 10.
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention() #多头自注意力 多头里包含Add&Norm
        self.dec_enc_attn = MultiHeadAttention()  #多头相互注意力-->对比EncoderLayer
        self.pos_ffn = PoswiseFeedForwardNet()    #前馈，其中包含Add&Norm

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
	
	# 自注意力
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
	
	# #交互注意力 Qx:dec_outputs Kx:enc_outputs Vx:enc_outputs
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
	
	# 前馈神经网络+残差
        dec_outputs = self.pos_ffn(dec_outputs)
	
        return dec_outputs, dec_self_attn, dec_enc_attn

# # 9. Decoder

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]::1x5:: [[5, 1, 2, 3, 4]]
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]::1x5x512
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, tgt_len, d_model] ::1x5x512

        # # get_attn_pad_mask 自注意力层的时候的pad 部分
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) #1x5x5全是false

        # # get_attn_subsequent_mask 这个做的是自注意层的mask部分，就是当前单词之后看不到，使用一个上三角为1的矩阵
	#       [[0, 1, 1, 1, 1],
	#        [0, 0, 1, 1, 1],
	#        [0, 0, 0, 1, 1],
	#        [0, 0, 0, 0, 1],
	#        [0, 0, 0, 0, 0]]	
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        # #两个矩阵相加，大于0的为1，不大于0的为0，为1的是mask的部分，在之后就会被fill到无限小
	#       [[[False,  True,  True,  True,  True],
	#         [False, False,  True,  True,  True],
	#	  [False, False, False,  True,  True],
	#	  [False, False, False, False,  True],
	#	  [False, False, False, False, False]]]	
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)   #


        # # 这个做的是交互注意力机制中的mask矩阵，enc的输入是k，我去看这个k里面哪些是pad符号，给到后面的模型；注意哦，我q肯定也是有pad符号，但是这里我不在意的，之前说了好多次了哈
	#       [[[False, False, False, False,  True],
	#         [False, False, False, False,  True],
	#         [False, False, False, False,  True],
	#         [False, False, False, False,  True],
	#         [False, False, False, False,  True]]]	
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask) #每一层的输出dec_outputs作为下一层的输入
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


# # 1. 从整体网路结构来看，分为三个部分：编码层，解码层，输出层
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()  # # 编码层 1x5--->>1x5x512
        self.decoder = Decoder()  # # 解码层 1x5x512 --->>1x5x512
	
	# 输出层 d_model 是我们解码层每个token输出的维度大小，之后会做一个 tgt_vocab_size 大小的softmax
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False) #1x5x512--->>1x5x7 相当于7分类    看论文流程图里这个linear后面应该还有个softmax
	
    def forward(self, enc_inputs, dec_inputs): #enc_inputs:[[1, 2, 3, 4, 0]] dec_inputs:[[5, 1, 2, 3, 4]] batch_size=1
	
        # # 这里有两个数据进行输入，一个是enc_inputs 形状为[batch_size, src_len]，主要是作为编码端的输入，一个dec_inputs，形状为[batch_size, tgt_len]，主要是作为解码端的输入

        # # enc_inputs作为输入 形状为[batch_size, src_len]，输出由自己的函数内部指定，想要什么指定输出什么，可以是全部tokens的输出，可以是特定每一层的输出；也可以是中间某些参数的输出；
        # # enc_outputs就是主要的输出，enc_self_attns这里没记错的是QK转置相乘之后softmax之后的矩阵值，代表的是每个单词和其他单词相关性；
        
        #输入1x5--->>输出1x5x512
        enc_outputs, enc_self_attns = self.encoder(enc_inputs) 

        # # dec_outputs 是decoder主要输出，用于后续的linear映射； dec_self_attns类比于enc_self_attns 是查看每个单词对decoder中输入的其余单词的相关性；dec_enc_attns是decoder中每个单词对encoder中每个单词的相关性；
        # 注意输入enc_inputs是怎么用？
	
	 #输入 dec_inputs:1x5 enc_inputs:1x5  enc_outputs:1x5x512 --->>输出 dec_outputs::1x5x512
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs) #为什么还需要enc_inputs？

        # # dec_outputs做映射到词表大小 输入 1x5x512--->> 输出1x5x7  因为输出有7种可能的word
        dec_logits = self.projection(dec_outputs) # dec_logits::[batch_size x src_vocab_size x tgt_vocab_size] ::1x5x7
	
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns



if __name__ == '__main__':

    # 句子的输入部分，
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E'] #前两个是做输入，第三个做输出，Transformer设计是用来解决翻译问题


    # Transformer Parameters
    # Padding Should be Zero
    # 构建词表
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5 # length of source
    tgt_len = 5 # length of target

    # 全局模型参数
    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention

    #模型初始化
    model = Transformer()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    #准备好训练数据-->需要2个输入 1个输出,此处按index编码
    #注意这里bach_size=1,只是为了演示方便，实际模型训练时bach_size不为1
    enc_inputs, dec_inputs, target_batch = make_batch(sentences) #enc_inputs: [[1, 2, 3, 4, 0]]   dec_inputs: [[5, 1, 2, 3, 4]] target_batch: [[1, 2, 3, 4, 6]]

    for epoch in range(20):
        optimizer.zero_grad()
	
	#模型训练
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs) #输入1x5和1x5 --->> 1x5x7 输出是7分类的概率值
	
	#计算损失
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
	
	#反向传播
        loss.backward()
        optimizer.step()



