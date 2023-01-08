#encoding=utf-8
import numpy as np
import pandas as pd
import jieba
import csv
from sklearn.feature_extraction.text import TfidfVectorizer

from snownlp import SnowNLP



#载入自定义词典
jieba.load_userdict("C:/Users/Eclipsa/Desktop/FYP/分词自定义词库/默认表情.txt")
jieba.load_userdict("C:/Users/Eclipsa/Desktop/FYP/分词自定义词库/东财小牛.txt")
jieba.load_userdict("C:/Users/Eclipsa/Desktop/FYP/分词自定义词库/财经小牛.txt")
jieba.load_userdict("C:/Users/Eclipsa/Desktop/FYP/分词自定义词库/菜刀豆.txt")
jieba.load_userdict("C:/Users/Eclipsa/Desktop/FYP/分词自定义词库/股市常用文本.txt")

#分词列
file_object1_train = pd.read_csv('C:/Users/Eclipsa/Desktop/FYP/000725_train.csv', encoding='utf-8') #一行行的读取内容
print(file_object1_train)
data_train = file_object1_train['Title']
print(data_train)

file_object1_test = pd.read_csv('C:/Users/Eclipsa/Desktop/FYP/002432_test.csv', encoding='utf-8') #一行行的读取内容
print(file_object1_test)
data_test = file_object1_test['标题']
print(data_test)


Rs1_train = [] #建立存储分词的列表
Rs1_test = []

# 创建停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open('C:/Users/Eclipsa/Desktop/FYP/分词自定义词库/停用词.txt',encoding='UTF-8').readlines()]
    return stopwords

#将训练集标题分词
for i in range(len(data_train)):
    result=[]
    sentence_depart = jieba.cut(data_train[i], cut_all=False, HMM=True) #fengci
    stopwords = stopwordslist() #ting
    for word in sentence_depart:
        if word not in stopwords:
                result.append(word)
    Rs1_train.append(result) #将该行分词写入列表形式的总分词列表

print(Rs1_train)

#将测试集标题分词
for i in range(len(data_test)):
    result=[]
    sentence_depart = jieba.cut(data_test[i], cut_all=False, HMM=True) #fengci
    stopwords = stopwordslist() #ting
    for word in sentence_depart:
        if word not in stopwords:
                result.append(word)
    Rs1_test.append(result) #将该行分词写入列表形式的总分词列表

print(Rs1_test)


#写入csv
file1_train=open('C:/Users/Eclipsa/Desktop/FYP/out1_train.txt','w',encoding='utf-8')
writer = csv.writer(file1_train)#定义写入格式
writer.writerows(Rs1_train)#按行写入
#file.write(str(Rs))
file1_train.close()

file1_test=open('C:/Users/Eclipsa/Desktop/FYP/out1_test.txt','w',encoding='utf-8')
writer = csv.writer(file1_test)#定义写入格式
writer.writerows(Rs1_test)#按行写入
#file.write(str(Rs))
file1_test.close()



# pre-processing techniques that can generate a numeric form from an input text
# from sklearn.feature_extraction.text import TfidfVectorizer


Rs2_train = [] #建立存储分词的列表
Rs2_test = []


def Tfidf(a):
  corpus = a
  vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b', encoding='utf-8', stop_words= None, max_df= 0.8, min_df=2, ngram_range= (2,2))
  itidf = vectorizer.fit_transform(corpus)
  it
  return itidf

# List 转化为 Array
for i in range(len(Rs1_train)):
    Rs1_train[i] = " ".join(Rs1_train[i])

# List 转化为 Array
for i in range(len(Rs1_test)):
    Rs1_test[i] = " ".join(Rs1_test[i])

X_all = Rs1_train + Rs1_test
len_train = len(Rs1_train)

t = Tfidf(X_all)

# TF-IDF
t1 = t[:len_train]
Rs2_train.append(t1) #将该行分词写入列表形式的总分词列表
t2 = t[len_train:]
Rs2_test.append(t2) #将该行分词写入列表形式的总分词列表


#写入txt
file_out2_train = open('C:/Users/Eclipsa/Desktop/FYP/out2_train.txt','w',encoding='utf-8')
writer = csv.writer(file_out2_train)#定义写入格式
writer.writerows(Rs2_train)#按行写入
#file.write(str(Rs))
file_out2_train.close()



#写入txt
file_out2_test = open('C:/Users/Eclipsa/Desktop/FYP/out2_test.txt','w',encoding='utf-8')
writer = csv.writer(file_out2_test)#定义写入格式
writer.writerows(Rs2_test)#按行写入
#file.write(str(Rs))
file_out2_test.close()



