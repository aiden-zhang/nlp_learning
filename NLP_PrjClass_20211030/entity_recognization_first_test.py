#coding:utf-8
# 来源：整理自深兰NLP项目课:第一次进行实体识别-->完成
# 功能：测试实体识别模型，利用模型预测
# 备注：

from collections import defaultdict
from operator import itemgetter
import numpy as np
import torch
import torch.nn.functional as F # pytorch 激活函数的类
import pickle as pk
import pandas as pd
from tqdm import tqdm
from model_zoo import * #模型定义

# 此处是加载对应的模型和配置文件
def load_model(model_name):
    parameter = pk.load(open('params_entityRecog.pkl','rb'))
    #     parameter['device'] = torch.device('cpu')
    # 因为bert模型需要加载他对应的config文件，因此此处进行了一定的区分
    if 'bert' in model_name:
        if 'speed' in model_name:
            model = eval(model_name.split('-')[0]+"(config,parameter).to(parameter['device'])")
        else:
            model = eval(model_name+"(config,parameter).to(parameter['device'])")
    else:
        model = eval(model_name+"(parameter).to(parameter['device'])") #这里必须要有模型定义，定义在model_zoo
    model.load_state_dict(torch.load(model_name+'.h5'))
    model.eval() 
    return model,parameter

# 将数组转成pytorch支持的输入
def list2torch(ins):
    return torch.from_numpy(np.array(ins))

# 此处和之前的数据预处理方式一致，不过这边是考虑bert有自带的字典因此，进行了一定的区分
def batch_yield(parameter,shuffle = True,isTrain = True,isBert = False):
    data_set = parameter['data_set']['train'] if isTrain else parameter['data_set']['dev']
    Epoch = parameter['epoch'] if isTrain else 1
    parameter['batch_size'] = 10
    for epoch in range(Epoch):
        # 每轮对原始数据进行随机化
        if shuffle:
            random.shuffle(data_set)
        inputs,targets = [],[]
        max_len = 0
        for items in tqdm(data_set):
            if not isBert:
                input = itemgetter(*items[0])(parameter['word2ind'])
                input = input if type(input) == type(()) else (input,0)
            else:
                input = tokenizer.convert_tokens_to_ids(items[0])
            target = itemgetter(*items[1])(parameter['key2ind'])
            target = target if type(target) == type(()) else (target,0)
            if len(input) > max_len:
                max_len = len(input)
            inputs.append(list(input))
            targets.append(list(target))
            if len(inputs) >= parameter['batch_size']:
                inputs = [i+[0]*(max_len-len(i)) for i in inputs]
                targets = [i+[-1]*(max_len-len(i)) for i in targets]
                yield list2torch(inputs),list2torch(targets),None,False
                inputs,targets = [],[]
                max_len = 0
        inputs = [i+[0]*(max_len-len(i)) for i in inputs]
        targets = [i+[-1]*(max_len-len(i)) for i in targets]
        yield list2torch(inputs),list2torch(targets),epoch,False
        inputs,targets = [],[]
        max_len = 0
    yield None,None,None,True

# 这边是模型评估
def eval_model(model_name):
    # 加载相应训练好的模型
    model,parameter = load_model(model_name)
    # 准备好待统计的内容
    count_table = {}
    # 根据是否文件名中包含bert字样判断是否为bert模型，决定使用哪个数据迭代器
    if 'bert' not in model_name:
        test_yield = batch_yield(parameter,shuffle = False,isTrain = False)
    else:
        test_yield = batch_yield(parameter,shuffle = False,isTrain = False,isBert = True)
    while 1:
        # 数据迭代
        inputs,targets,_,keys = next(test_yield)
        if not keys:
            # 获取相应模型数据的预测结果
            pred = model(inputs.long().to(parameter['device']))
            # 因为crf模型和直接用softmax模型推理方面有一定的区别，因此根据crf模型或者softmax模型进行区分
            if 'crf' in model_name:
                # crf模型需要对内容进行解码，得到相应的结果
                predicted_index = np.array(model.crf.decode(pred))
                targets = targets.numpy()#.long().to(parameter['device'])
            else:
                # softmax模型直接使用softmax区最大值
                predicted_prob,predicted_index = torch.max(F.softmax(pred, 1), 1)
                predicted_index = predicted_index.reshape(inputs.shape)
                targets = targets.long().to(parameter['device'])
            # 此处注意，回忆一下精确度和召回率的定义；
            # 精确度是，大致可以描述为，判断正确的正例/预测中总共判断正例的数量
            # 召回率是，大致可以描述为，判断正确的正例/实际中总共正例的数量
            # 由此可以得到以下处理的方法：
            # 提前准备好tp，
            right = (targets == predicted_index)
            for i in range(1,parameter['output_size']):
                if i not in count_table:
                    count_table[i] = {
                    'pred':len(predicted_index[(predicted_index == i) & (targets != -1)]), # i标签下的，tp+fp，预测总正例
                    'real':len(targets[targets == i]),# i标签下的，tp+fn，实际总正例
                    'common':len(targets[right & (targets == i)])# i标签下的tp
                    }
                else:
                    count_table[i]['pred'] += len(predicted_index[predicted_index == i])
                    count_table[i]['real'] += len(targets[targets == i])
                    count_table[i]['common'] += len(targets[right & (targets == i)])
        else:
            break
    count_pandas = {}
    # 获取对应标签中文名，和相应统计值，从1开始，为了过滤标签O的统计
    name,count = list(parameter['key2ind'].keys())[1:],list(count_table.values())
    for ind,i in enumerate(name):
        # 'B-*','I-*','E-*','S-*'都可以用'-'分割，合并同样标签的内容
        i = i.split('-')[1]
        # 综合统计
        if i in count_pandas:
            count_pandas[i][0] += count[ind]['pred']
            count_pandas[i][1] += count[ind]['real']
            count_pandas[i][2] += count[ind]['common']
        else:
            count_pandas[i] = [0,0,0]
            count_pandas[i][0] = count[ind]['pred']
            count_pandas[i][1] = count[ind]['real']
            count_pandas[i][2] = count[ind]['common']
    # 计算总数
    count_pandas['all'] = [sum([count_pandas[i][0] for i in count_pandas]),
                      sum([count_pandas[i][1] for i in count_pandas]),
                      sum([count_pandas[i][2] for i in count_pandas])]
    name = count_pandas.keys()
    count_pandas = pd.DataFrame(count_pandas.values())
    count_pandas.columns = ['pred','real','common']
    # 基于tp、tp+fn、tp+fp计算相应的p、r以及计算f1；回忆一下f1计算公式：2pr/(p+r)，fn：(1+b^2)/(b^2)*(pr)/(p+r)，f1好处？
    count_pandas['p'] = count_pandas['common']/count_pandas['pred']
    count_pandas['r'] = count_pandas['common']/count_pandas['real']
    count_pandas['f1'] = 2*count_pandas['p']*count_pandas['r']/(count_pandas['p']+count_pandas['r'])
    count_pandas.index = list(name)
    return count_pandas

def keyword_predict(input):
    input = list(input)
#     input_id = tokenizer.convert_tokens_to_ids(input)
    input_id = itemgetter(*input)(parameter['word2ind']) #汉字转数字索引
    input_id = input_id if type(input_id) == type(()) else (input_id,0)
    
    predict = model.crf.decode(model(list2torch([input_id]).long().to(parameter['device'])))[0] #预测结果是标签的数字索引
    predict = itemgetter(*predict)(parameter['ind2key']) #数字索引转标签
    print(predict)
    keys_list = []
    for ind,i in enumerate(predict):
        if i == 'O':
            continue
        if i[0] == 'S':
            if not(len(keys_list) == 0 or keys_list[-1][-1]):
                del keys_list[-1]
            keys_list.append([input[ind],[i],[ind],True])
            continue
        if i[0] == 'B':
            if not(len(keys_list) == 0 or keys_list[-1][-1]):
                del keys_list[-1]
            keys_list.append([input[ind],[i],[ind],False])
            continue
        if i[0] == 'I':
            if len(keys_list) > 0 and not keys_list[-1][-1] and \
            keys_list[-1][1][0].split('-')[1] == i.split('-')[1]:
                keys_list[-1][0] += input[ind]
                keys_list[-1][1] += [i]
                keys_list[-1][2] += [ind]
            else:
                if len(keys_list) > 0:
                    del keys_list[-1]
            continue
        if i[0] == 'E':
            if len(keys_list) > 0 and not keys_list[-1][-1] and \
            keys_list[-1][1][0].split('-')[1] == i.split('-')[1]:
                keys_list[-1][0] += input[ind]
                keys_list[-1][1] += [i]
                keys_list[-1][2] += [ind]

                keys_list[-1][3] = True  
#keys_list:[['浙商银行', ['B-company', 'I-company', 'I-company', 'E-company'], [0, 1, 2, 3], True], ['叶老桂', ['B-name', 'I-name', 'E-name'], [9, 10, 11], True]]
            else:
                if len(keys_list) > 0:
                    del keys_list[-1]
            continue
    return keys_list


if __name__=='__main__':
    
    if False: #模型评价
        ret=eval_model('bilstm_crf')
        print(ret)
    else: #预测
        model,parameter = load_model('bilstm_crf')
        # tokenizer = tokenizer_class.from_pretrained("prev_trained_model")
        model = model.to(parameter['device'])
        
        test_text = '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言'
        ret=keyword_predict(test_text)   
        print(ret) #[['浙商银行', ['B-company', 'I-company', 'I-company', 'E-company'], [0, 1, 2, 3], True], ['叶老桂', ['B-name', 'I-name', 'E-name'], [9, 10, 11], True]]
    