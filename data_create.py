#coding:utf-8
#整理自：数据准备.ipynb

# 获取所有诗人名

import pandas as pd
import numpy as np
import os

if __name__=='__main__':


#2  构建两种结构化数据（树状结构、自组织结构）+++++++++++++++++++++++++++++++++++
    #2.1  处理各科知识点=================================
    dicts = {}
    error_num = 0
    error = []
    uppath = '原始数据/'
    for path in ['高中地理/','高中历史/','高中生物/','高中政治/']:
        dicts[path[:-1]] = {}
        for i in os.listdir(uppath+path+'origin/'): #'生产活动与地域联系.csv'
            dicts[path[:-1]][i.split('.')[0]] = []
            data = pd.read_csv(uppath+path+'origin/'+i).item
            for j in data:
                j = j.split('\n')
                if len(j) == 4:
                    questions = j[1].replace('知识点：','') #去掉多余文字得到问题
                    kg = j[3].split(',') #所属知识点
                    dicts[path[:-1]][i.split('.')[0]].append([questions,kg])
                elif len(j) > 4:
                    questions = []
                    questions_key = False
                    kg = []
                    kg_key = False
                    for tmp in j:
                        if "[题目]" == tmp:
                            questions_key = True
                            continue
                        if "[知识点：]" == tmp:
                            questions_key = False
                            kg_key = True
                            continue
                        if questions_key:
                            questions.append(tmp)
                        if kg_key:
                            kg.append(tmp)
                    if len(kg) > 0:
                        questions = ''.join(questions).replace('知识点：','')
                        kg = ''.join(kg).split(',')
                        dicts[path[:-1]][i.split('.')[0]].append([questions,kg])
                    else:
                        error_num += 1
                        error.append([path,i,j])
                else:
                    error_num += 1
                    error.append([path,i,j])
                    
    # 构建数据类型为《父科目，子科目，知识点，题目》
    out_path = '数据生成/'
    os.mkdir(out_path) if not os.path.exists(out_path) else 1
    data_new = []
    for km1 in dicts:
        for km2 in dicts[km1]:
            for data in dicts[km1][km2]:
                for kg in data[1]:
                    data_new.append([km1,km2,kg,data[0]])
                    
    data_new = pd.DataFrame(data_new)
    data_new.columns = ['km1','km2','kg','question'] #添加列标题
    
    if not os.path.exists(out_path+'km.csv'):
        data_new.to_csv(out_path+'km.csv',index = False) #存到:'数据生成/km.csv中'
    else:
        pass
    
    #2.2  构建知识图谱的数据===============================
    #  2.2.1  处理树状结构数据--------------------------
    # 处理科目表，树级结构
    
    def long_num_str(data):
        data = str(data)+'\t'
        return data
    
    # path = 'data_combine/'
    data = pd.read_csv(out_path+'km.csv')
    entity = None
    for i in data:
        names = list(set(eval('data.'+i)))
        ids = [hash(j) for j in names]
        labels = [i for j in names]
        if entity is None:
            entity = pd.DataFrame(np.array([ids,names,labels]).transpose())
        else:
            entity = entity.append(pd.DataFrame(np.array([ids,names,labels]).transpose()))
    
    entity.columns = [':ID','name',':LABEL']
    entity[':ID'] = entity[':ID'].map(long_num_str)
    entity.to_csv(out_path+'entity_km.csv',index = False) #存到:'数据生成/entity_km.csv中'
    
    
    relation = None
    for i in [[data.keys()[tmp],data.keys()[tmp+1]] for tmp in range(len(data.keys())-1)]:
        tmp = data[i]
        tmp = tmp.drop_duplicates(subset=i[1] , keep='first',inplace=False)
    #     print(len(tmp))
        start = [hash(j) for j in eval('tmp.'+i[0])]
        end = [hash(j) for j in eval('tmp.'+i[1])]
        name = [i[0]+'->'+i[1] for j in range(len(tmp))]
        if relation is None:
            relation = pd.DataFrame(np.array([start,name,end,name]).transpose())
        else:
            relation = relation.append(pd.DataFrame(np.array([start,name,end,name]).transpose()))
    
    relation.columns = [':START_ID','name',':END_ID',':TYPE']
    relation[':START_ID'] = relation[':START_ID'].map(long_num_str)
    relation[':END_ID'] = relation[':END_ID'].map(long_num_str)
    relation.to_csv(out_path+'relation_km.csv',index = False)     #存到:'数据生成/relation_km.csv中'
    
#2.2.2  处理自组织结构数据-------------------------------
    # 处理古诗表，自组织结构
    path = '原始数据/古诗/'
    poems = pd.read_csv(path+'诗句.csv')
    poets = pd.read_csv(path+'poets.csv')
    poets = dict(zip(poets.poet,poets.introduce))
    introduce = []
    for i in poems.author:
        if i in poets:
            introduce.append(poets[i])
        else:
            introduce.append(np.nan)
    poems['introduce'] = introduce
    
    # 统计最大tag长度,为7
    # max([len(eval(i)) for i in poems['tag'] if type(i) == type('')])
    
    tag = []
    for i in poems['tag']:
        if type(i) == type(''):
            i = eval(i)
            i += [np.nan]*(7-len(i))
        else:
            i = [np.nan]*7
        tag.append(i)
        
    poems[['tag-'+str(i) for i in range(7)]] = pd.DataFrame(tag)
    poems = poems[['author','introduce','title','content','type','translate','class']+['tag-'+str(i) for i in range(7)]]    
    
    entity = None
    for i in poems:
        names = list(set(eval('poems["'+i+'"]')))
        ids = [hash(j) for j in names]
        if 'tag-' in i:
            i = 'tag'
        labels = [i for j in names]
        if entity is None:
            entity = pd.DataFrame(np.array([ids,names,labels]).transpose())
        else:
            entity = entity.append(pd.DataFrame(np.array([ids,names,labels]).transpose()))
    
    entity.columns = [':ID','name',':LABEL']
    entity[':ID'] = entity[':ID'].map(long_num_str)
    entity.to_csv(out_path+'entity_poems.csv',index = False) #存到:'数据生成/entity_poems.csv中'
    
    relation = None
    for i in [['author','title'],['author','introduce'],['title','content'],['title','translate'],
             ['type','title'],['class','title']]+[['tag-'+str(i),'title'] for i in range(7)]:
        tmp = poems[i]
        tmp = tmp.dropna()
        start = [hash(j) for j in eval('tmp["'+i[0]+'"]')]
        end = [hash(j) for j in eval('tmp["'+i[1]+'"]')]
        if 'tag-' in i[0]:
            i[0] = 'tag'
        name = [i[0]+'->'+i[1] for j in range(len(tmp))]
        if relation is None:
            relation = pd.DataFrame(np.array([start,name,end,name]).transpose())
        else:
            relation = relation.append(pd.DataFrame(np.array([start,name,end,name]).transpose()))
    
    relation.columns = [':START_ID','name',':END_ID',':TYPE']
    relation[':START_ID'] = relation[':START_ID'].map(long_num_str)
    relation[':END_ID'] = relation[':END_ID'].map(long_num_str)
    relation.to_csv(out_path+'relation_poems.csv',index = False) #存到:'数据生成/relation_poems.csv中'
    
    
    
#2.3  生成合并后的实体表和关系表=====================================================================================
    path = '数据生成/'
    file_list = ['km','poems']
    entity_list = [pd.read_csv(path+'entity_'+i+'.csv') for i in file_list]
    relation_list = [pd.read_csv(path+'relation_'+i+'.csv') for i in file_list]
    
    sum([len(i) for i in entity_list]),sum([len(i) for i in relation_list])
    
    entity_com = entity_list[0].append(entity_list[1])#.append(entity_list[2])
    relation_com = relation_list[0].append(relation_list[1])#.append(relation_list[2])
    entity_com = entity_com.drop_duplicates(subset=':ID' , keep='first',inplace=False)
    entity_com.to_csv(path+'entity.csv') #存到:'数据生成/entity.csv中'
    
    relation_com = relation_com.drop_duplicates(subset=[':START_ID', 'name', ':END_ID', ':TYPE'] , keep='first',inplace=False)
    relation_com.to_csv(path+'relation.csv') #存到:'数据生成/relation.csv中'
    len(relation_com)
    
    entity_com
    
    relation_com
    
#3  构建实体识别数据集-基于规则构建+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #3.1  基于知识图谱准备待识别的实体===============================
    # 待识别的实体主要有km1,km2,kg;author,title,tag,type,class
    import pandas as pd
    path = '数据生成/'
    data = pd.read_csv(path+'entity.csv')
    data = data[data[':LABEL'].isin(['km1','km2','kg','author','title','tag','type','class'])]
    data = data[['name',':LABEL']]
    data.columns = ['name','label']
    data.to_csv('entity.csv',index = False)    #存到:'./entity.csv中' 与先前的entity.csv的不同是去掉了ID:这一列，原LABL:改为lable
    
  #3.2  基于上述实体及预设好的提问回答，生成意图识别和实体识别的数据=================
    data = pd.read_csv('entity.csv')
    data_new = {}
    for i in range(len(data)):
        i = data.loc[i]
        name = i['name'].replace('（','').replace('）','').replace('【','').replace('】','').replace('“','').replace('”','').replace('《','').replace('》','').replace('(','').replace(')','').replace('_','')
        name_list = [m for j in name.split('·') for k in j.split('、') for q in k.split('、') for p in q.split('，') for m in p.split('・') for n in m.split('/')]
        label = i.label
        for name in name_list:
            # 过滤一些较长的实体
            if len(name) > 10:
                continue
            if label not in data_new:
                data_new[label] = [name]
            else:
                data_new[label].append(name)
            
    [[len(data_new[i]),i] for i in data_new],sum([len(data_new[i]) for i in data_new])
    
    
    from random import choice
    def build_seq(data,label,seq1,seq2,intent,mode = 1,num = 0,output = None):
        '''
        data：代表前面处理的实体列表
        label：代表该实体所代表的的标签如km1
        seq1：代表实体前置的语料
        seq2：代表实体后置的语料
        intent：代表该生成语料的意图
        mode：1代表遍历，2代表随机提取
        num：如果mode=2，num则代表随机生成num个句子
        '''
        seqs_intent,labels_intent,seqs_ner,labels_ner = [],[],[],[]
        seqs_intent.append(list(seq1)+[label]+list(seq2))
        labels_intent.append(intent)
        if mode == 1:
            for i in data[label]:
                seqs_ner.append(list((seq1+'%s'+seq2)%(i)))
                if len(i) > 2:
                    labels_ner.append(['O']*len(seq1)+["B-"+label]+(len(i)-2)*['I-'+label]+['E-'+label]+['O']*len(seq2))
                if len(i) == 2:
                    labels_ner.append(['O']*len(seq1)+["B-"+label]+['E-'+label]+['O']*len(seq2))
                if len(i) < 2:
                    labels_ner.append(['O']*len(seq1)+['S-'+label]+['O']*len(seq2))
        if mode == 2:
            for i in range(num):
                i = choice(data[label])
                seqs_ner.append(list((seq1+'%s'+seq2)%(i)))
                if len(i) > 2:
                    labels_ner.append(['O']*len(seq1)+["B-"+label]+(len(i)-2)*['I-'+label]+['E-'+label]+['O']*len(seq2))
                if len(i) == 2:
                    labels_ner.append(['O']*len(seq1)+["B-"+label]+['E-'+label]+['O']*len(seq2))
                if len(i) < 2:
                    labels_ner.append(['O']*len(seq1)+['S-'+label]+['O']*len(seq2))
        if output is None:
            return seqs_intent,labels_intent,seqs_ner,labels_ner
        else:
            output[0] += seqs_intent
            output[1] += labels_intent
            output[2] += seqs_ner
            output[3] += labels_ner
    
    def create_dataset(seqs,out_put,label,intent,keys = 1,num = 100):
        seq1_list,seq2_list = seqs
        if keys == 1:
            for seq1 in seq1_list:
                for seq2 in seq2_list:
                    build_seq(data_new,
                        label = label,
                        seq1 = seq1,seq2 = seq2,
                        intent = intent,mode = 1, num = 10,
                     output = out_put)
        else:
            iffirst = True
            mode = 1
            for seq1 in seq1_list:
                for seq2 in seq2_list:
                    if not iffirst:
                        mode = 2
                    iffirst = False
                    build_seq(data_new,
                            label = label,
                            seq1 = seq1,seq2 = seq2,
                            intent = intent,mode = mode, num = num,
                     output = out_put)
                    
    x_intent,y_intent,x_ner,y_ner = [],[],[],[]
    
    output = [x_intent,y_intent,x_ner,y_ner]
    
    label = 'km1'
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['有哪些重要的课程','有哪些重要的课','什么课比较重要','需要学什么课',
                 '有哪些课']
    create_dataset([seq1_list,seq2_list],output,label,0)
    
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['有哪些重要的知识点','有哪些知识点需要注意','什么知识点比较重要',
                 '需要学什么那些知识点', '有哪些知识点','涉及到哪些知识点','包含哪些知识点']
    create_dataset([seq1_list,seq2_list],output,label,1)
    
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['有哪些重要的例题需要掌握','有哪些例题需要注意','什么例题需要掌握',
                  '有哪些例题']
    create_dataset([seq1_list,seq2_list],output,label,2)
    
    
    label = 'km2'
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['是哪个学科的课程','是什么学科的']
    create_dataset([seq1_list,seq2_list],output,label,3)
    
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['有哪些重要的知识点','有哪些知识点需要注意','什么知识点比较重要',
                 '需要学什么那些知识点', '有哪些知识点','涉及到哪些知识点','包含哪些知识点']
    create_dataset([seq1_list,seq2_list],output,label,4)
    
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['有哪些重要的例题需要掌握','有哪些例题需要注意','什么例题需要掌握',
                 '有哪些例题']
    create_dataset([seq1_list,seq2_list],output,label,5)
    
    label = 'kg'
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['是什么学科的','是哪个学科的知识点']
    create_dataset([seq1_list,seq2_list],output,label,6,keys = 2)
    
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['是什么课程的','是哪个课程需要学习的知识点','是哪门课的']
    create_dataset([seq1_list,seq2_list],output,label,7,keys = 2)
    
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['有哪些重要的例题需要掌握','有哪些例题需要注意','什么例题需要掌握',
                 '有哪些例题']
    create_dataset([seq1_list,seq2_list],output,label,8,keys = 2)
    
    ## 针对古诗的处理
    label = 'author'
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['写过哪些诗句啊','有哪些诗','有哪些有名的诗句',
                 ]
    create_dataset([seq1_list,seq2_list],output,label,9,keys = 2,num = 500)
    
    
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['是谁啊','这个的生平怎么样','有哪些经历',
                 ]
    create_dataset([seq1_list,seq2_list],output,label,10,keys = 2,num = 500)
    
    label = 'title'
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['是谁写的啊'
                 ]
    create_dataset([seq1_list,seq2_list],output,label,11,keys = 2,num = 500)
    
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['的诗句有哪些'
                 ]
    create_dataset([seq1_list,seq2_list],output,label,12,keys = 2,num = 500)
    
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['的翻译是'
                 ]
    create_dataset([seq1_list,seq2_list],output,label,13,keys = 2,num = 500)
    
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['是什么类型的古诗'
                 ]
    create_dataset([seq1_list,seq2_list],output,label,14,keys = 2,num = 500)
    
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['出现在几年级的课程'
                 ]
    create_dataset([seq1_list,seq2_list],output,label,15,keys = 2,num = 500)
    
    label = 'tag'
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['的古诗有哪些'
                 ]
    create_dataset([seq1_list,seq2_list],output,label,16,keys = 2,num = 100)
    
    
    label = 'class'
    seq1_list = ['我们','','请问我们']
    seq2_list = ['学习的古诗有哪些'
                 ]
    create_dataset([seq1_list,seq2_list],output,label,17,keys = 1,num = 100)
    
    len(x_intent),len(y_intent),len(x_ner),len(y_ner)
    
    import pickle as pk
    pk.dump([x_intent,y_intent,x_ner,y_ner],open('data-intent1-ner.pkl','wb'))
    
    import pickle as pk
    [x_intent,y_intent,x_ner,y_ner] = pk.load(open('data-intent1-ner.pkl','rb'))
    print('所构建的意图识别的数据有：',len(x_intent))
    print('所构建意图识别数据的样例：\n',x_intent[:2],'\n',y_intent[:2])
    print('所构建实体识别数据的样例：\n',x_ner[:2],'\n',y_ner[:2])    
    
#4  构建意图识别数据+++++++++++++++++++++++++++++++++++++++++++
    
    # 针对语料数据的处理
    corpus = [list(i.strip()) for i in open('data/corpus/conversation_test.txt','r',
                                            encoding = 'utf-8').readlines()][::2]
    corpus_new = {}
    for i in corpus:
        if str(i) not in corpus_new:
            corpus_new[str(i)] = 1
    corpus = [eval(i) for i in list(corpus_new.keys())]
    len(corpus),corpus[:2]
    
    # 针对于机器人个人属性数据的处理
    import xml.etree.ElementTree as et
    data = et.parse('data/robot_template.xml').findall('temp')
    data = [j.text.replace('(.*)','').replace('.*','') for i in data for j in i.find('question').findall('q')]
    data_new = [i for i in data if '[' not in i]
    data_need_deal = [
        '[怎么|如何|怎样|用|什么方式][称呼|叫]',
        '[说|讲|介绍]你的[名字|姓名]',
        '[Ash|小智]是谁',
        '你是[Ash|小智]吗',
        '你[的|有]性别[是什么]',
        '你是[男|女|美女|帅哥|小姐姐|小哥哥]',
        '你[爸|父亲|爹|创造者|造物主|主人|主子|设计人|设计者]是谁',
        '是谁[生|创造][了|的]你',
         '你[妈|母亲|娘]是谁',
        '你[有|的][男|女|妻子|丈夫|老婆|老公|爱人|夫人|相公|媳妇|朋友]',
        '你是[单身|结婚]',
        '你[的|有][职业|工作][是][什么|啥]',
        '你[是]干什么[工作|活|职业]的',
        '[说|介绍|讲]你的[职业|工作]']
    data_need_deal_new = []
    for i in data_need_deal:
        i = i.split(']')
        tmp = []
        for j in i:
            if '[' not in j:
                if len(j) != 0:
                    tmp.append(j)
            else:
                for k in j.split('['):
                    if len(k) != 0:
                        tmp.append(k)
        data_need_deal_new.append(tmp)
        
    def build(n,m = 0,tmp = '',res = []):
        if m < len(n):
            if '|' in n[m]:
                for i in n[m].split('|'):
                    build(n,m+1,tmp+i,res)
            else:
                build(n,m+1,tmp+n[m],res)
        if m >= len(n):
            res.append(tmp)
    
    for i in data_need_deal_new:
        res = []
        build(i,res = res)
        data_new += res
        
    data_new = [list(i) for i in data_new]
        
    len(data_new),data_new[:2]
    
    # combine and save
    data_x = x_intent+data_new+corpus
    data_y = [0]*len(x_intent)+[1]*len(data_new)+[2]*len(corpus)
    pk.dump([data_x,data_y],open('data-intent0.pkl','wb')),data_x[:2],data_y[:2]    
    
    #5  trash
    x_intent,y_intent,x_ner,y_ner = [],[],[],[]
    
    def create_dataset(seq1_list,seq2_list,out_put):
        for seq1 in seq1_list:
            for seq2 in seq2_list:
                build_seq(data_new,
                    label = 'km1',
                    seq1 = seq1,seq2 = seq2,
                    intent = 0,mode = 1, num = 10,
                 output = out_put)
    
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['有哪些重要的课程','有哪些重要的课','什么课比较重要','需要学什么课',
                 '有哪些课']
    for seq1 in seq1_list:
        for seq2 in seq2_list:
            build_seq(data_new,
                    label = 'km1',
                    seq1 = seq1,seq2 = seq2,
                    intent = 0,mode = 1, num = 10,
             output = [x_intent,y_intent,x_ner,y_ner])
    
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['有哪些重要的知识点','有哪些知识点需要注意','什么知识点比较重要',
                 '需要学什么那些知识点', '有哪些知识点','涉及到哪些知识点','包含哪些知识点']
    for seq1 in seq1_list:
        for seq2 in seq2_list:
            build_seq(data_new,
                    label = 'km1',
                    seq1 = seq1,seq2 = seq2,
                    intent = 1,mode = 1, num = 10,
             output = [x_intent,y_intent,x_ner,y_ner])
            
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['有哪些重要的例题需要掌握','有哪些例题需要注意','什么例题需要掌握',
                  '有哪些例题']
    for seq1 in seq1_list:
        for seq2 in seq2_list:
            build_seq(data_new,
                    label = 'km1',
                    seq1 = seq1,seq2 = seq2,
                    intent = 2,mode = 1, num = 10,
             output = [x_intent,y_intent,x_ner,y_ner])
            
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['是哪个学科的课程','是什么学科的']
    for seq1 in seq1_list:
        for seq2 in seq2_list:
            build_seq(data_new,
                    label = 'km2',
                    seq1 = seq1,seq2 = seq2,
                    intent = 3,mode = 1, num = 10,
             output = [x_intent,y_intent,x_ner,y_ner])
            
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['有哪些重要的知识点','有哪些知识点需要注意','什么知识点比较重要',
                 '需要学什么那些知识点', '有哪些知识点','涉及到哪些知识点','包含哪些知识点']
    for seq1 in seq1_list:
        for seq2 in seq2_list:
            build_seq(data_new,
                    label = 'km2',
                    seq1 = seq1,seq2 = seq2,
                    intent = 4,mode = 1, num = 10,
             output = [x_intent,y_intent,x_ner,y_ner])
    
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['有哪些重要的例题需要掌握','有哪些例题需要注意','什么例题需要掌握',
                 '有哪些例题']
    for seq1 in seq1_list:
        for seq2 in seq2_list:
            build_seq(data_new,
                    label = 'km2',
                    seq1 = seq1,seq2 = seq2,
                    intent = 5,mode = 1, num = 10,
             output = [x_intent,y_intent,x_ner,y_ner])
            
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['是什么学科的','是哪个学科的知识点']
    iffirst = True
    mode = 1
    num = 10
    for seq1 in seq1_list:
        for seq2 in seq2_list:
            if not iffirst:
                mode = 2
                num = 100
            iffirst = False
            build_seq(data_new,
                    label = 'kg',
                    seq1 = seq1,seq2 = seq2,
                    intent = 6,mode = mode, num = num,
             output = [x_intent,y_intent,x_ner,y_ner])
            
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['是什么课程的','是哪个课程需要学习的知识点','是哪门课的']
    iffirst = True
    mode = 1
    num = 10
    for seq1 in seq1_list:
        for seq2 in seq2_list:
            if not iffirst:
                mode = 2
                num = 100
            iffirst = False
            build_seq(data_new,
                    label = 'kg',
                    seq1 = seq1,seq2 = seq2,
                    intent = 7,mode = mode, num = num,
             output = [x_intent,y_intent,x_ner,y_ner])
            
    seq1_list = ['老师','','请问','你好']
    seq2_list = ['有哪些重要的例题需要掌握','有哪些例题需要注意','什么例题需要掌握',
                 '有哪些例题']
    iffirst = True
    mode = 1
    num = 10
    for seq1 in seq1_list:
        for seq2 in seq2_list:
            if not iffirst:
                mode = 2
                num = 100
            iffirst = False
            build_seq(data_new,
                    label = 'kg',
                    seq1 = seq1,seq2 = seq2,
                    intent = 8,mode = mode, num = num,
             output = [x_intent,y_intent,x_ner,y_ner])
            
    len(x_intent),len(y_intent),len(x_ner),len(y_ner)    