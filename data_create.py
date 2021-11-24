#coding:utf-8
#整理自：数据准备.ipynb

# 获取所有诗人名

import pandas as pd
import numpy as np

if __name__=='__main__':

    # 清洗诗人简介的数据
    data = np.array(pd.read_csv('nlp/nlp/poets.csv'))
    data_new = []
    for i in data:
        if '百度百科' not in i[0]:
            introduce = eval(i[1])[0].replace('<meta name="description" content="','').replace('...">','').replace(' ','')
            if introduce[-1] not in ['。','！','？']:
                introduce = '。'.join([j for j in introduce.split('。')][:-1])
                if len(introduce) != 0:
                    introduce += '。'
            if len(introduce) != 0:
                data_new.append([i[0],introduce])
        
    data_new = pd.DataFrame(data_new)
    data_new.columns = ['poet','introduce']
    data_new.to_csv('原始数据/古诗/poets.csv',index = False)    