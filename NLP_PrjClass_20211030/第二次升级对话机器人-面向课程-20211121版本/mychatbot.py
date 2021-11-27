'''
场景1：基于对话机器人自身属性的对话

场景2：基于已有语料进行匹配的对话

场景3：基于已有知识库进行搜索的对话

场景4：基于互联网进行对话

场景5：基于bert+seq2seq进行生成式对话

扩展1：基于Scrapy爬虫框架，爬取更多知识库

扩展2：使用实体识别、word2vec等丰富对话系统的泛化能力

扩展3：使用flask进行服务的包装

扩展4：使用pyqt完成对话机器人客户端
'''

import logging
from config import *
import xml.etree.ElementTree as et
from random import choice
import re

class BaseLayer:
    """基础父类"""

    def __init__(self, log=True):
        self.logger = logging.getLogger()
        if not log:
            self.close_log()

    def close_log(self):
        self.logger.setLevel(logging.ERROR)

    def print_log(self, msg):
        self.logger.warning(msg)

    def search_answer(self, question):
        pass

class template(BaseLayer):
    '''
    基于机器人自身属性的回答
    '''
    def __init__(self):
        super(template,self).__init__()
        self.template = self.load_template_file()
        self.robot_info = self.load_robot_info()
        self.temps = self.template.findall('temp')
        self.default_answer = self.template.find('default')
        self.exceed_answer = self.template.find('exceed')

    def load_template_file(self):
        data = et.parse(TEMPLATE_PATH)
        return data

    def load_robot_info(self):
        rebot_info = self.template.find('robot_info')
        rebot_info_dict = {}
        for info in rebot_info:
            rebot_info_dict[info.tag] = info.text
        return rebot_info_dict

    def search_answer(self, question):
        match_temp = None
        flag = None
        for temp in self.temps:
            qs = temp.find('question').findall('q')
            for q in qs:
                res = re.search(q.text,question)
                if res:
                    match_temp = temp
                    flag = True
                    break
            if flag:
                break
        if flag:
            a_s = choice([i.text for i in match_temp.find('answer').findall('a')])
            answer = a_s.format(**self.robot_info)
            return answer
        else:
            return None


from method import build_inverse,load_seq_qa
from collections import Counter
import numpy as np
import jieba

class CorpusSearch(BaseLayer):
    def __init__(self):
        super(CorpusSearch,self).__init__()
        self.question_list,self.answer_list = load_seq_qa()
        self.inverse = build_inverse()
        self.THRESHOLD = 0.7

    def cosine_sim(self, a, b):
        """计算两个句子的 余弦相似度"""
        a_words = Counter(a)
        b_words = Counter(b)
        # 建立两个句子的 字典 vocabulary
        all_words = b_words.copy()
        all_words.update(a_words - b_words)
        all_words = set(all_words)
        # 生成句子向量
        a_vec, b_vec = list(), list()
        for w in all_words:
            a_vec.append(a_words.get(w, 0))
            b_vec.append(b_words.get(w, 0))
        # 计算余弦相似度值
        a_vec = np.array(a_vec)
        b_vec = np.array(b_vec)
        a__ = np.sqrt(np.sum(np.square(a_vec)))
        b__ = np.sqrt(np.sum(np.square(b_vec)))
        cos_sim = np.dot(a_vec, b_vec) / (a__ * b__)
        return round(cos_sim, 4)

    def search_answer(self, question):
        # 分词后，各词都出现在了哪些文档中
        search_list = list()
        q_words = jieba.lcut(question)
        for q_word in q_words:
            index = self.inverse.get(q_word, list())
            search_list += index
        if not search_list:
            return None
        # 统计包含问句中词汇次数最多的 3 个文档
        count_list = Counter(search_list)
        count_list = count_list.most_common(3)
        result_sim = list()
        for i, _ in count_list:
            q = self.question_list[i]
            sim = self.cosine_sim(q_words, q)
            result_sim.append((i, sim))
        # 根据两两的余弦相似度选出最相似的问句
        result = max(result_sim, key=lambda x: x[1])
        # 如果相似度低于阈值，则返回 None
        if result[1] > self.THRESHOLD:
            answer = ''.join(self.answer_list[result[0]])
            return answer
        else:
            return None



import requests
class InterNet(BaseLayer):
    def __init__(self):
        super(InterNet,self).__init__()
        self.print_log('InterNet layer is ready.')

    def search_answer(self,question):
        url = 'https://api.ownthink.com/bot?appid=xiaosi&userid=user&spoken='
        try:
            text = requests.post(url+question).json()
            if 'message' in text and text['message'] == 'success':
                return text['data']['info']['text']
            else:
                return None
        except:
            return None





class Generate(BaseLayer):
    """
        生成式对话
    """



# test = InterNet()
# test = CorpusSearch()
# test = template()
# while 1:
    # print(test.search_answer(input('请输入->')))