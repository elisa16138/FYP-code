from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def Tfidf(a):
  corpus = ['u' + " ".join(s)]
  vectorizer = CountVectorizer(stop_words=None)
  transformer = TfidfTransformer(norm='l1')
  tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

  words = vectorizer.get_feature_names()  # 所有文本的关键字
  weight = tfidf.toarray()  # 对应的tfidf矩阵
  [print(element, end=' ') for element in words]
  print()
  print(weight)
  print(vectorizer.vocabulary_[u"打板"])
  print(tfidf[0, vectorizer.vocabulary_[u"打板"]])


file = open('C:/Users/Eclipsa/Downloads/12/out.txt').read().split('\n')
for i in file:
  x = file(i)


