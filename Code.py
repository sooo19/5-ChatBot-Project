import pandas as pd
import gzip
from gensim.models.doc2vec import TaggedDocument
from gensim.models import doc2vec
from nltk.corpus import stopwords  
from nltk import word_tokenize  
from nltk.stem import WordNetLemmatizer
import numpy as np


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


import tensorflow as tf
import os
#dataset download
'''
dataset = tf.keras.utils.get_file( fname='qa_Video_Games.json.gz',
                                 origin = "http://jmcauley.ucsd.edu/data/amazon/qa/qa_Video_Games.json.gz", # 링크주소 
                                  extract = True)
'''
df = getDF('qa_Video_Games.json.gz')

stop_words = stopwords.words('english')

tokened_q = [word_tokenize(question.lower()) for question in df['question']]

lemmatizer = WordNetLemmatizer()
lemmed_questions = [[lemmatizer.lemmatize(word) for word in doc] for doc in tokened_q]
questions = [[w for w in doc if not w in stop_words] for doc in lemmed_questions]


index_questions = []
for i in range(len(df)):
    index_questions.append([questions[i], i ])
    
tagged_questions = [TaggedDocument(d, [int(c)]) for d, c in index_questions]

import multiprocessing
cores = multiprocessing.cpu_count()
d2v_faqs = doc2vec.Doc2Vec(
                                vector_size=1000,
                                hs=1,
                                negative=0,
                                dm=0,
                                dbow_words = 1,
                                min_count = 10,
                                workers = cores,
                                seed=0,
                                epochs=20
                                )
d2v_faqs.build_vocab(tagged_questions)
d2v_faqs.train(tagged_questions,
                total_examples = d2v_faqs.corpus_count,
                epochs = d2v_faqs.epochs
                                )

df_sample=df[['question']]

print("해당 카테고리의 질문: ")
print(df_sample)


print("질문을 입력하세요: ")
test_string = input()
question=test_string
tokened_test_string = word_tokenize(test_string)
lemmed_test_string = [lemmatizer.lemmatize(word) for word in tokened_test_string]
test_string = [w for w in lemmed_test_string if not w in stop_words]
print('test string : {}'.format(test_string))

topn = 5

test_vector = d2v_faqs.infer_vector(test_string)
result = d2v_faqs.docvecs.most_similar([test_vector], topn=topn)

select_question = []
for i in range(topn):
    print("{}위. 정확도: {}, {}".format(i+1, result[i][1], df['question'][result[i][0]] ))
    select_question.append(df['question'][result[i][0]])
print("\n")

print("입력한 질문: ")
for x in select_question:
    list1_x=x.split()
    list2_question=question.split()
    if list1_x == list2_question:
        print("Question: ", x)
    else:
        print("이 질문이 아닙니다")
    
print("\nAnswer: {}".format(df['answer'][result[0][0]]))


print("제품의 asin 번호: {}".format(df['asin'][result[0][0]]))
asin_num=df['asin'][result[0][0]]



import tensorflow as tf
import os

#dataset download
'''
dataset = tf.keras.utils.get_file( fname='meta_Vedio_Games.json.gz',
                                 origin = "http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Video_Games.json.gz", # 링크주소 
                                  extract = True)
'''

'''
dataset = tf.keras.utils.get_file( fname='review_Video_Games.json.gz',
                                  origin = "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Video_Games.json.gz", # 링크주소 
                                  extract = True)

basic_dir = os.path.dirname(dataset) 
print(basic_dir)
'''

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df_review = getDF('review_Video_Games.json.gz')

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df_meta = getDF('meta_Vedio_Games.json.gz')

asin = asin = df['asin'][result[0][0]]
answer_asin = np.where(df_meta['asin'].values == asin)

print(answer_asin)

print("질문한 제품에 대한 정보\n")

for i in range(0,100):
    x=answer_asin[i]
    print("1. 가격 정보")
    print(df1['price'][x])
    print("\n")
    
    print("2. 제품 이미지")
    print(df1['image'][x])
    print("\n")
    
    print("3. 제품 브랜드")
    print(df1['brand'][x])
    print("\n")
    
    print("4. 제품 리뷰")
    print(df_review['reviewText'])