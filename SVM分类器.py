#encoding=utf-8
import numpy as np
import pandas as pd
import jieba
import csv
from sklearn.feature_extraction.text import TfidfVectorizer


#利用额外爬的数据手工标注后进行模型的训练与选择

#载入自定义词典
jieba.load_userdict("C:/Users/Eclipsa/Desktop/FYP/分词自定义词库/默认表情.txt")
jieba.load_userdict("C:/Users/Eclipsa/Desktop/FYP/分词自定义词库/东财小牛.txt")
jieba.load_userdict("C:/Users/Eclipsa/Desktop/FYP/分词自定义词库/财经小牛.txt")
jieba.load_userdict("C:/Users/Eclipsa/Desktop/FYP/分词自定义词库/菜刀豆.txt")
jieba.load_userdict("C:/Users/Eclipsa/Desktop/FYP/分词自定义词库/股市常用文本.txt")
# 创建停用词列表

def stopwordslist():
    stopwords = [line.strip() for line in open('C:/Users/Eclipsa/Desktop/FYP/分词自定义词库/停用词.txt',encoding='UTF-8').readlines()]
    return stopwords


#导入训练数据集
file_object= pd.read_csv('C:/Users/Eclipsa/Desktop/FYP/000725_train.csv', encoding='utf-8') #一行行的读取内容#
x_train = file_object['Title']
y_train = file_object['Sentiment']
print(x_train.shape)
print(y_train.shape)
x_train = np.array(x_train)

#将测试集标题分词
Rs1 = [] #建立存储分词的列表
for i in range(len(x_train)):
    result=[]
    sentence_depart = jieba.cut(x_train[i], cut_all=False, HMM=True) #fengci
    stopwords = stopwordslist() #ting
    for word in sentence_depart:
        if word not in stopwords:
                result.append(word)
    Rs1.append(result) #将该行分词写入列表形式的总分词列表



#写入csv
#file_out1=open('C:/Users/Eclipsa/Desktop/FYP/out1_train_naivebayessin.txt','w',encoding='utf-8')
#writer = csv.writer(file_out1)#定义写入格式
#writer.writerows(Rs1)#按行写入
#file.write(str(Rs))
#file_out1.close()

# pre-processing techniques that can generate a numeric form from an input text
# from sklearn.feature_extraction.text import TfidfVectorizer


Rs2= [] #建立存储分词的列表

#定义TF-IDF模型 正则化、特征值设置
def Tfidf(a):
  corpus = a
  vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b',
                               encoding='utf-8', stop_words= None, max_df= 0.8, min_df=2, ngram_range= (1,4))
  itidf = vectorizer.fit_transform(corpus)
  return itidf

# List 转化为 Array
for i in range(len(Rs1)):
    Rs1[i] = " ".join(Rs1[i])

#特征向量化
t = Tfidf(Rs1)
Rs2.append(t) #将该行分词写入列表形式的总分词列表


#写入txt
#file_out2 = open('C:/Users/Eclipsa/Desktop/FYP/out2_train_naivebayessin.txt','w',encoding='utf-8')
#writer = csv.writer(file_out2)#定义写入格式
#writer.writerows(Rs2)#按行写入
#file.write(str(Rs))
#file_out2.close()



from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
#SVM 分类器参数调优
params = {'kernel':['linear', 'rbf'], 'C':[0.1, 1, 5, 10]}
svc = SVC(probability = True, random_state = 0)
clf = GridSearchCV(svc, param_grid = params, scoring = 'accuracy', cv = 20, n_jobs = -1)
clf.fit(t, y_train)
print('Best score: {}'.format(clf.best_score_))
print('Best parameters: {}'.format(clf.best_params_))