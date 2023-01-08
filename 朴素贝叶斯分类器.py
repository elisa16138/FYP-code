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

file_object_test = pd.read_csv('C:/Users/Eclipsa/Desktop/FYP/002432_test.csv', encoding='utf-8') #一行行的读取内容
print(file_object_test.shape)
x_test = file_object_test['标题']
print(x_test.shape)
data_test = np.array(x_test)

#将测试集标题分词
Rs1_train = [] #建立存储分词的列表
for i in range(len(x_train)):
    result=[]
    sentence_depart = jieba.cut(x_train[i], cut_all=False, HMM=True) #fengci
    stopwords = stopwordslist() #ting
    for word in sentence_depart:
        if word not in stopwords:
                result.append(word)
    Rs1_train.append(result) #将该行分词写入列表形式的总分词列表

#将训练集标题分词
Rs1_test = [] #建立存储分词的列表
for i in range(len(x_test)):
    result=[]
    sentence_depart = jieba.cut(x_test[i], cut_all=False, HMM=True) #fengci
    stopwords = stopwordslist() #ting
    for word in sentence_depart:
        if word not in stopwords:
                result.append(word)
    Rs1_test.append(result) #将该行分词写入列表形式的总分词列表


# pre-processing techniques that can generate a numeric form from an input text
# from sklearn.feature_extraction.text import TfidfVectorizer


Rs2_train = [] #建立存储分词的列表
Rs2_test = []

#定义TF-IDF模型 正则化、特征值设置
def Tfidf(a):
  corpus = a
  vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b',
                               encoding='utf-8', stop_words= None, max_df= 0.8, min_df=2, ngram_range= (1,4))
  itidf = vectorizer.fit_transform(corpus)
  itidf_1 = itidf.toarray()
  return itidf_1


# List 转化为 Array
for i in range(len(Rs1_train)):
    Rs1_train[i] = " ".join(Rs1_train[i])

for i in range(len(Rs1_test)):
    Rs1_test[i] = " ".join(Rs1_test[i])


X_all = Rs1_train + Rs1_test
len_train = len(Rs1_train)

# 这一步有点慢，去喝杯茶刷会儿微博知乎歇会儿...
t_all = Tfidf(X_all)
# 恢复成训练集和测试集部分
t_train = t_all[:len_train]
t_test = t_all[len_train:]


from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB as MNB

# 设定grid search的参数
params = {'alpha': [0.1, 0.4, 0.5, 0.6, 0.55, 0.7, 0.9, 1.0]}
nb = MNB()
clf = GridSearchCV(nb, param_grid = params, scoring = 'accuracy', cv = 20, n_jobs = -1)
clf.fit(t_train, y_train)
print('Best score: {}'.format(clf.best_score_))
print('Best parameters: {}'.format(clf.best_params_))

nb_best = MNB( alpha=clf.best_params_['alpha'])
