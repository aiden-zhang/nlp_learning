
#coding:utf-8
##预处理
#适当清洗语料+加载语料
from config import *
import re
import jieba
def load_seq_qa():
    q_list,a_list = [],[]
    with open(CORPUS_PATH,'r',encoding = 'utf-8') as f:
        for ind, i in enumerate(f):
            #print(f"ind:{ind},i{i}")
            i = jieba.lcut(i.strip())
            if ind % 2 == 0:
                q_list.append(i)#偶数行是答案
            else:
                a_list.append(i)
    return q_list,a_list

print("问题:\n",load_seq_qa()[0][:10],'\n\n',"答案:\n",load_seq_qa()[1][:10])

#针对提问，构建词库，将切割后的词转换为index，nlp常用手段
def build_vocab():
    q,_ = load_seq_qa()
    word_dict = set([j for i in q for j in i]) #set可以去重？
    #print(word_dict) #所有question分词后放入一个set
    word_dict = dict(zip(word_dict,range(len(word_dict))))#不重复的所有的词映射到index
    return word_dict
    
print(build_vocab())

def build_word_embeding():
    q,_ = load_seq_qa()
    word_dict = build_vocab()
    word_embeding = {}
    for w in word_dict.keys():
        word_embeding[w] = []
    for ind,qs in enumerate(q):#qs及其对应的ind
        for w in qs: #w是词，qs是句子
            #print(f"w:{w}\n qs:{qs}")
            
            #把qs对应序号作为其中所有词w的的embeding，这个序号不同于word_dict中的序号
            word_embeding[w].append(ind)
    return word_embeding #返回的是词及其在其中出现的所有句子的index

from collections import Counter
import numpy as np
import jieba

class corpusSearch():
    def __init__(self):
        self.q,self.a = load_seq_qa()
        self.word_embeding = build_word_embeding()
        self.threshold = 0.7

    def cosine_sim(self,a,b):
        a_count = Counter(a)
        b_count = Counter(b)
        a_vec = []
        b_vec = []
        all_word = set(a+b)
        #这边类似构建了一个句子的embeding
        for i in all_word:
            a_vec.append(a_count.get(i,0))
            b_vec.append(b_count.get(i,0))
        #计算余弦相似
        a_vec = np.array(a_vec)
        b_vec = np.array(b_vec)
        cos = sum(a_vec*b_vec)/(sum(a_vec*a_vec)**0.5)/(sum(b_vec*b_vec)**0.5)
        return round(cos,4)

    def search_answer(self,question):#找到相似度最高的句子
        search_list = []
        q_words = jieba.lcut(question)
        for q_word in q_words:
            index = self.word_embeding.get(q_word,list())#找出q_word对应的索引list
            search_list += index
        print(search_list) 
        if len(search_list) == 0:
            return None
        #挑选前3个
        count_list = Counter(search_list)
        count_list = count_list.most_common(3)
        print(count_list)
        res_list = []
        for i,_ in count_list:
            q = self.q[i]
            sim = self.cosine_sim(q,q_words)
            if sim > self.threshold:
                res_list.append([sim,self.a[i]])
        res_list.sort()
        if len(res_list) == 0:
            return None
        else:
            return ''.join(res_list[-1][1])

test = corpusSearch()
while 1:
    question = input('请输入->')
    if question == ':':
        break
    print(test.search_answer(question))
    
q,_=load_seq_qa()
print(q[97:100])



from py2neo import Graph
from pyhanlp import *
from random import choice


#知识图谱的初步应用
class GraphSearch():
    def __init__(self):
        self.graph = Graph("http://localhost:7474", username="neo4j", password="123456")
        self.iswho_sql = "profile match p=(n)<-[r]-(b) where n.name='%s' return n.name,r.name,b.name"
        self.isrelation_sql = "profile match p=(n)<-[r]-(b) where n.name=~'.*%s' and b.name=~'.*%s' return n.name,r.name,b.name" 
    
    def search_answer(self,question):
        #使用HanLP进行词性判断
        sentence = HanLP.parseDependency(question)
        #后续可以替换成自己训练的模块首先针对句意进行分析，其次针对目标实体进行提取；但主要也是针对业务场景进行分析和处理
        seg = {}
        res_combine = ''
        for word in sentence.iterator(): 
        ##只处理nr名词：人，v动词，n名词，针对进行提问进行词性分析
            if word.POSTAG[0] == 'n' or word.POSTAG in ['v','r']:
                if word.POSTAG not in seg:
                    seg[word.POSTAG] = [word.LEMMA]
                else:
                    seg[word.POSTAG].append(word.LEMMA)
        print(seg,'*'*10)
        #简单基于词性和内容判断是否为目标句式'A是谁'以此使用知识图谱进行回答
        if 'v' in seg and '是' in seg['v']:
            if 'r' in seg and 'nr' in seg and '谁' in seg['r']:
                for person in seg['nr']:
                    res = self.graph.run(self.iswho_sql%(person)).data()
                    res_combine = []
                    for i in res[:10]:
                        res_combine.append('%s是:%s%s'%(i['n.name'],i['b.name'],i['r.name']))
                return choice(res_combine)
        #基于词性和内容判断是否为目标句式'A和B的关系'以此使用知识图谱进行回答
        if 'n' in seg and '关系' in seg['n']:
            if len(seg['nr']) == 2:
                print(seg,'*1'*10)
                res1 = self.graph.run(self.isrelation_sql%(seg['nr'][1],seg['nr'][0])).data()
                if res1 != []:
                    res_combine = seg['nr'][0]+'的'+res1[0]['r.name']+'是'+seg['nr'][1]
                    return res_combine
                res2 = self.graph.run(self.isrelation_sql%(seg['nr'][0],seg['nr'][1])).data()
                if res2 != []:
                    res_combine = seg['nr'][1]+'的'+res2[0]['r.name']+'是'+seg['nr'][0]
                    return res_combine
        if res_combine == '':
            return None
        
test = GraphSearch()
while 1:
    print(test.search_answer(input('请输入->')))        