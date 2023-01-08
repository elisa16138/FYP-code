import requests
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
proxyHost = "u5568.5.tn.16yun.cn"
proxyPort = "6441"
proxyUser = "16WIEGLN"
proxyPass = "XXX"
proxyMeta = "http://%(user)s:%(pass)s@%(host)s:%(port)s" % {
     "host" : proxyHost,
     "port" : proxyPort,
     "user" : proxyUser,
     "pass" : proxyPass,
 }

# 设置 http和https访问都是用HTTP代理
proxies = {
     "http"  : proxyMeta,
     "https" : proxyMeta,
 }
error=[]
lst = []
headers={'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'}

for page in range(1,8819):
    url='https://guba.eastmoney.com/list,000725,f_{}.html'.format(page)
    try:
        html=requests.get(url=url,headers=headers,timeout=5)
        soup=BeautifulSoup(html.text,'html.parser')
        sj_lst=soup.select('#articlelistnew div')
        for i in sj_lst[1:]:
            try:
                data={}
                #data['股票id'] = 'id:'+str(sid)
                data['阅读'] = i.select('span')[0].text
                data['评论'] = i.select('span')[1].text
                data['标题'] = i.select('span')[2].text
                data['作者'] = i.select('span')[3].text
                data['最后更新'] = i.select('span')[4].text
                data['链接'] = 'https:'+i.select('a')[0]['href']
                lst.append(data)
            except:
                pass
        print(page)
    except:
        error.append(page)

for page in error:
    url='https://guba.eastmoney.com/list,000725,f_{}.html'.format(page)
    try:
        html=requests.get(url=url,headers=headers,timeout=5)
        soup=BeautifulSoup(html.text,'html.parser')
        sj_lst=soup.select('#articlelistnew div')
        for i in sj_lst[1:]:
            try:
                data={}
                #data['股票id'] = 'id:'+str(sid)
                data['阅读'] = i.select('span')[0].text
                data['评论'] = i.select('span')[1].text
                data['标题'] = i.select('span')[2].text
                data['作者'] = i.select('span')[3].text
                data['最后更新'] = i.select('span')[4].text
                data['链接'] = 'https:'+i.select('a')[0]['href']
                lst.append(data)
            except:
                pass
        print(page)
    except:
        pass

result = pd.DataFrame(lst)
result.to_excel('000725.xlsx',index=None)
result.to_csv('000725.csv',index=None,encoding='utf-8-sig')