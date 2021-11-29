
2021.10.30开班的深兰NLP项目课，这是中NLP项目课第第一阶段，后续还有第二阶段
===================================================================

## 文件结构

#text_classification_first_train.py
	意图识别训练代码

#text_classification_first_test.py   
	意图识别模型评价与预测的代码
	
#entity_recognization_first_train.py 
	实体识别训练代码

#entity_recognization_first_test.py  
	实体识别模型评价与预测代码
	
#第一次构建一个简易的对话机器人

	第一次实现了对话机器人，主要内容：
	
		1.基于robot_template.xml文件内容，实现了基于机器人自身属性提问与答案搜索；
		
		2.基于对话语料库conversation_test.txt，对于用户提出的问题，按照相似性匹配原理从语料库找出最相近的问题，其在语料库中的答案近似作为用户提问的答案；
		
		3.基于第一课用entity.csv和relationship.csv构建的知识图谱存与neo4j中，对于用户针对该知识图谱的提问，从中搜索出答案；
		
		4.将以上三种问答的答案检索通过flask放在服务端，用户的提问通过client端向server请求，server找出对应问题答案后发给client。
		
#第二次升级对话机器人-面向课程-20211121版本

	第二次实现对话机器人，主要内容：
	
		1.利用原有的高中课程相关知识，及通过爬虫获取到的诗人简介构建出数据集entity.csv和relation.csv,基于文件中的实体和关系通过neo4j构建知识图谱；
		
		2.基于知识图谱构建出语料库data-intent1-ner.pkl，基于闲聊+自身属性+知识图谱构建出语料库data-intent0.pkl；
		
		3.训练两个意图识别模型，一个命名实体识别模型，具体方法：
		
			①利用data-intent0.pkl训练意图识别模型0，训练后的模型用于预测用户提问属于闲聊or自身属性or识图谱，若属于知识图谱则会利用意图识别模型1，再次划分；
			
			②利用data-intent1-ner.pkl训练意图识别模型1，该模型预测结果的十几个标签代表知识图谱的十几个问题类型，每种类型对应neo4j搜索语句不通，因此要分开；
			③利用data-intent1-ner.pkl训练ner模型，该模型会针对用户问题进行加工，替换调其中的实体生成entity_list，对结果进行rebuildin生成new_question，再将new_question送入意图识别模型0,进行分类，预测结果是intent0，若intent0指向知识图谱的提问，则再将new_question送入意图识别模型1,预测出类别intent1,这代表意图识别问题的一个细分方向，最后将参数entity_list和intent1传入意图识别搜索接口，从数据库中搜索数据作为问题的结果返回给用户。
			
		4.将3的功能、各种答案搜索或模型预测接口放在用flask搭建的服务端。
		
说明：这次对话机器人的问题都是基于规则人工做出了的，因此语料规模有限，这也是后续可以优化的点，再有就是ner模型在整个项目中的功能还有有些不清楚，后续还需要仔细研究！
			
