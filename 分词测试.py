import jieba

#载入自定义词典
jieba.load_userdict("C:/Users/Eclipsa/Desktop/FYP/分词自定义词库/默认表情.txt")
jieba.load_userdict("C:/Users/Eclipsa/Desktop/FYP/分词自定义词库/东财小牛.txt")
jieba.load_userdict("C:/Users/Eclipsa/Desktop/FYP/分词自定义词库/财经小牛.txt")
jieba.load_userdict("C:/Users/Eclipsa/Desktop/FYP/分词自定义词库/菜刀豆.txt")
jieba.load_userdict("C:/Users/Eclipsa/Desktop/FYP/分词自定义词库/股市常用文本.txt")


# encoding=utf-8
import jieba

#汉药都不行，苗药有能怎样，快跑，快跑
seg_list = jieba.cut("汉药都不行，苗药有能怎样，快跑，快跑", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("医药长坡赛道大PK，业绩不输医药女神！看富国生物医药", cut_all=False, HMM=True)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("医药长坡赛道大PK，业绩不输医药女神！看富国生物医药", cut_all=False, HMM=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式


#翰宇药业里的不良资本还没有清理干净，估计暂时是横盘阴跌，见好就收吧！
seg_list = jieba.cut("翰宇药业里的不良资本还没有清理干净，估计暂时是横盘阴跌，见好就收吧！", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("翰宇药业里的不良资本还没有清理干净，估计暂时是横盘阴跌，见好就收吧！", cut_all=False, HMM=True)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("翰宇药业里的不良资本还没有清理干净，估计暂时是横盘阴跌，见好就收吧！", cut_all=False, HMM=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

