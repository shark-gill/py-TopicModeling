#%%
# ===== 01_Import Packages =====
import numpy as np
import pandas as pd
import re
re.compile('<title>(.*)</title>')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import konlpy # pip install konlpy
from konlpy.tag import Mecab

#%%
# ===== 02_Dataset Load=====
rawdata = pd.read_csv('/home/lhshrk/py-TopicModeling/data/202209-12.csv', encoding='cp949')
# rawdata.head() / Tabel 확인
# len(rawdata) / 49
# documents = rawdata['내용'].values.tolist()
# print(documents)

def extract_word(text): # 한글만 출력하는 함수
    hangul = re.compile('[^가-힣]') 
    result = hangul.sub(' ', text) 
    return result

# print("Before Extract", rawdata['내용'][30])
# print("After Extract", extract_word(rawdata['내용'][30]))

rawdata['내용'] = rawdata['내용'].apply(lambda x:extract_word(x))

# rawdata

#%%
# ===== 03-1_Preprocessing - Spacing =====
from pykospacing import Spacing

spacing = Spacing()
print("Before Fixing : ",rawdata['내용'][30])
print("After Fixing : ", spacing(rawdata['내용'][30]))

#%%
# ===== 03-2_Preprocessing - Tokenization =====
from konlpy.tag import Mecab # pip install python-mecab-ko

tagger = Mecab()

words = " ".join(rawdata['내용'].tolist())
words = tagger.morphs(words)
words

#%%
# ===== 03-3_Preprocessing - Stopwords Remove =====
remove_one_word = [x for x in words if len(x)>1 or x=="닉"]
len(remove_one_word)


with open('/home/lhshrk/py-TopicModeling/data/stopwords.txt', 'r', encoding='cp949') as f:
    list_file = f.readlines()
stopwords = list_file[0].split(",")
remove_stopwords = [x for x in remove_one_word if x not in stopwords]
    # len(remove_stopwords)

#%%
# ===== 04_Kewords frequency Analysis & list to dict =====
from collections import Counter
frequent = Counter(remove_one_word)
top_freq = dict(frequent.most_common())
len(top_freq)
# result_df = pd.DataFrame.from_dict(top_freq, orient='index')
# result_df

#%%
# ===== 05_Save File =====
import csv

with open('/home/lhshrk/py-TopicModeling/result/result_whole.csv', 'w', encoding='cp949') as file:
    writer = csv.writer(file)
    for key, value in top_freq.items():
       writer.writerow([key, value])


## Source
# 1) 키워드 분석 방법 DOI: https://www.youtube.com/watch?v=5P6nG8xHKbU&t=1147s
# 2) ★ Mecab 설치 방법 DOI: https://velog.io/@shchoice/KoNLPy-%EC%84%A4%EC%B9%98-Ubuntu
# 3) Mecab 설치 방법 DOI: https://konlpy-ko.readthedocs.io/ko/v0.4.3/install/#ubuntu
# 4) Mecab 설치 방법2 DOI: https://heytech.tistory.com/395
# 5) JAVA Install DOI: https://davelogs.tistory.com/71
# 6) pykospacing Install DOI: https://github.com/haven-jeon/PyKoSpacing
# 7) ★ 키워드 분석 방법 & CLASS/FUNC Exam★ DOI: https://haystar.tistory.com/11
# %%
