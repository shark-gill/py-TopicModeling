#%%
# ! /home/lhshrk/.pyenv/shims/python3
# ! Not Virtual Env
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
rawdata = pd.read_csv('/home/lhshrk/py-TopicModeling/data/2023_07-09_key_freq.csv', encoding='cp949')
rawdata.head() # Tabel 확인
len(rawdata)
documents = rawdata['내용'].values.tolist()
print(documents)

#%%
# ===== 03-1_Preprocessing - Regex(Kor) Extraction =====
def extract_word(text): # 한글만 출력하는 함수
    hangul = re.compile('[^가-힣]') 
    result = hangul.sub(' ', str(text)) 
    return result

#%%
print("Before Extract", rawdata['내용'][31])
print("After Extract", extract_word(rawdata['내용'][31]))

for i in rawdata.columns[2:12]:
    rawdata[i] = rawdata[i].apply(lambda x:extract_word(x))

rawdata

#%%
# ===== 03-2_Preprocessing - Spacing =====
from pykospacing import Spacing

spacing = Spacing()
for i in rawdata.columns[2:12]:
    spacing(rawdata[i])

print("Before Fixing : ", rawdata[2:12][:])
print("After Fixing : ", spacing(rawdata[2:12][:]))

#%%
# ===== 03-3_Preprocessing - Tokenization =====
from konlpy.tag import Mecab # pip install python-mecab-ko

tagger = Mecab()

column_name = rawdata.columns[2:12].to_list()
column_name
#%%

word1 = " ".join(rawdata[column_name[0]].tolist())
word1 = tagger.morphs(word1)

word2 = " ".join(rawdata[column_name[1]].tolist())
word2 = tagger.morphs(word2)

word3 = " ".join(rawdata[column_name[2]].tolist())
word3 = tagger.morphs(word3)

word4 = " ".join(rawdata[column_name[3]].tolist())
word4 = tagger.morphs(word4)

word5 = " ".join(rawdata[column_name[4]].tolist())
word5 = tagger.morphs(word5)

word6 = " ".join(rawdata[column_name[5]].tolist())
word6 = tagger.morphs(word6)

word7 = " ".join(rawdata[column_name[6]].tolist())
word7 = tagger.morphs(word7)

word8 = " ".join(rawdata[column_name[7]].tolist())
word8 = tagger.morphs(word8)

word9 = " ".join(rawdata[column_name[8]].tolist())
word9 = tagger.morphs(word9)

word10 = " ".join(rawdata[column_name[9]].tolist())
word10 = tagger.morphs(word10)


#%%
# ===== 03-4_Preprocessing - Stopwords Remove =====
remove_one_word1 = [x for x in word1 if len(x)>1 or x=="닉"]
remove_one_word2 = [x for x in word2 if len(x)>1 or x=="닉"]
remove_one_word3 = [x for x in word3 if len(x)>1 or x=="닉"]
remove_one_word4 = [x for x in word4 if len(x)>1 or x=="닉"]
remove_one_word5 = [x for x in word5 if len(x)>1 or x=="닉"]
remove_one_word6 = [x for x in word6 if len(x)>1 or x=="닉"]
remove_one_word7 = [x for x in word7 if len(x)>1 or x=="닉"]
remove_one_word8 = [x for x in word8 if len(x)>1 or x=="닉"]
remove_one_word9 = [x for x in word9 if len(x)>1 or x=="닉"]
remove_one_word10 = [x for x in word10 if len(x)>1 or x=="닉"]


#%%
# ★ 이거 실행하면 오류 뜸!!! ★
# 따라서, 건너뛰기!
with open('/home/lhshrk/py-TopicModeling/data/stopwords.txt', 'r', encoding='cp949') as f:
    list_file = f.readlines()
stopwords = list_file[0].split(",")
remove_stopword1 = [x for x in remove_one_word1 if x not in stopwords]
remove_stopword2 = [x for x in remove_one_word2 if x not in stopwords]
remove_stopword3 = [x for x in remove_one_word3 if x not in stopwords]
remove_stopword4 = [x for x in remove_one_word4 if x not in stopwords]
remove_stopword5 = [x for x in remove_one_word5 if x not in stopwords]
remove_stopword6 = [x for x in remove_one_word6 if x not in stopwords]
remove_stopword7 = [x for x in remove_one_word7 if x not in stopwords]
remove_stopword8 = [x for x in remove_one_word8 if x not in stopwords]
remove_stopword9 = [x for x in remove_one_word9 if x not in stopwords]
remove_stopword10 = [x for x in remove_one_word10 if x not in stopwords]
    # len(remove_stopwords)

#%%
# ===== 04_Kewords frequency Analysis & list to dict =====
from collections import Counter

frequent1 = Counter(remove_one_word1)
top_freq1 = dict(frequent1.most_common())
frequent2 = Counter(remove_one_word2)
top_freq2 = dict(frequent2.most_common())
frequent3 = Counter(remove_one_word3)
top_freq3 = dict(frequent3.most_common())
frequent4 = Counter(remove_one_word4)
top_freq4 = dict(frequent4.most_common())
frequent5 = Counter(remove_one_word5)
top_freq5 = dict(frequent5.most_common())
frequent6 = Counter(remove_one_word6)
top_freq6 = dict(frequent6.most_common())
frequent7 = Counter(remove_one_word7)
top_freq7 = dict(frequent7.most_common())
frequent8 = Counter(remove_one_word8)
top_freq8 = dict(frequent8.most_common())
frequent9 = Counter(remove_one_word9)
top_freq9 = dict(frequent9.most_common())
frequent10 = Counter(remove_one_word10)
top_freq10 = dict(frequent10.most_common())

#%%
top_freq1
# len(top_freq)
# top_freq
# result_df = pd.DataFrame.from_dict(top_freq, orient='index')
# result_df

#%%
# ===== 05_Save File =====
import csv
    
with open('/home/lhshrk/py-TopicModeling/result/result_01.csv', 'w', encoding='cp949') as file:
    writer = csv.writer(file)
    for key, value in top_freq1.items():
       writer.writerow([key, value])

with open('/home/lhshrk/py-TopicModeling/result/result_02.csv', 'w', encoding='cp949') as file:
    writer = csv.writer(file)
    for key, value in top_freq2.items():
       writer.writerow([key, value])

with open('/home/lhshrk/py-TopicModeling/result/result_03.csv', 'w', encoding='cp949') as file:
    writer = csv.writer(file)
    for key, value in top_freq3.items():
       writer.writerow([key, value])

with open('/home/lhshrk/py-TopicModeling/result/result_04.csv', 'w', encoding='cp949') as file:
    writer = csv.writer(file)
    for key, value in top_freq4.items():
       writer.writerow([key, value])

with open('/home/lhshrk/py-TopicModeling/result/result_05.csv', 'w', encoding='cp949') as file:
    writer = csv.writer(file)
    for key, value in top_freq5.items():
       writer.writerow([key, value])

with open('/home/lhshrk/py-TopicModeling/result/result_06.csv', 'w', encoding='cp949') as file:
    writer = csv.writer(file)
    for key, value in top_freq6.items():
       writer.writerow([key, value])

with open('/home/lhshrk/py-TopicModeling/result/result_07.csv', 'w', encoding='cp949') as file:
    writer = csv.writer(file)
    for key, value in top_freq7.items():
       writer.writerow([key, value])

with open('/home/lhshrk/py-TopicModeling/result/result_08.csv', 'w', encoding='cp949') as file:
    writer = csv.writer(file)
    for key, value in top_freq8.items():
       writer.writerow([key, value])

with open('/home/lhshrk/py-TopicModeling/result/result_09.csv', 'w', encoding='cp949') as file:
    writer = csv.writer(file)
    for key, value in top_freq9.items():
       writer.writerow([key, value])

with open('/home/lhshrk/py-TopicModeling/result/result_10.csv', 'w', encoding='cp949') as file:
    writer = csv.writer(file)
    for key, value in top_freq10.items():
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
