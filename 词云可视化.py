import os
import sys
import json
import numpy as np
from PIL import Image
from scipy.misc import imread
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from wordcloud import WordCloud,ImageColorGenerator,STOPWORDS

reload(sys)
sys.setdefaultencoding('utf-8')

# 自定义颜色列表
color_list = ['#CD853F', '#DC143C', '#00FF7F', '#FF6347', '#8B008B', '#00FFFF', '#0000FF', '#8B0000', '#FF8C00',
              '#1E90FF', '#00FF00', '#FFD700', '#008080', '#008B8B', '#8A2BE2', '#228B22', '#FA8072', '#808080']



def simpleWC3(sep=' ',back='black',freDictpath='data_fre.json',savepath='res.png'):
    '''
    词云可视化Demo【自定义字体的颜色】
    '''
    #基于自定义颜色表构建colormap对象
    colormap=colors.ListedColormap(color_list)
    try:
        with open(freDictpath,'w') as f:
            data=f.readlines()
            data_list=[one.strip().split(sep) for one in data if one]
        fre_dict={}
        for one_list in data_list:
            fre_dict[unicode(one_list[0])]=int(one_list[1])
    except:
        fre_dict=freDictpath
    wc=WordCloud(font_path='font/simhei.ttf',#设置字体  #simhei
                background_color=back,  #背景颜色
                max_words=1300,  #词云显示的最大词数
                max_font_size=120,  #字体最大值
                colormap=colormap,  #自定义构建colormap对象
                margin=2,width=1800,height=800,random_state=42,
                prefer_horizontal=0.5)  #无法水平放置就垂直放置
    wc.generate_from_frequencies(fre_dict)
    plt.figure()
    plt.imshow(wc)
    plt.axis("off")
    wc.to_file(savepath)



if __name__ == '__main__':
    text=open('C:/Users/Eclipsa/Downloads/12/out.txt',encoding='UTF-8')
    word_list=text.split()
    fre_dict={}
    for one in word_list:
        if one in fre_dict:
            fre_dict[one]+=1
        else:
            fre_dict[one]=1

    simpleWC3(sep=' ',back='black',freDictpath=fre_dict,savepath='simpleWC3.png'
