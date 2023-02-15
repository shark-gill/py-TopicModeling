#%%
##영어 텍스트##
f = open("key_freq.txt",'rt',encoding='utf-8')
lines = f.readlines()
line = []
for i in range(len(lines)):
    line.append(lines[i])
f.close()

#%%
##특수문자 제거##
import re

compile = re.compile("\W+")
for i in range(len(line)):

    a = compile.sub(" ",line[i])
    line[i] = a.lower()
line

#%%
import nltk
##처음 한번만 실행 후 주석처리##
nltk.download('all')
nltk.download('wordnet')
nltk.download('stopwords')
##############################
from nltk.corpus import stopwords
stop_word_eng = set(stopwords.words('english'))
line = [i for i in line if i not in stop_word_eng]

#%%
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
ps_stemmer = PorterStemmer()
token = RegexpTokenizer('[\w]+')
result = [token.tokenize(i) for i in line]
middle_result= [r for i in result for r in i]
final_result = [ps_stemmer.stem(i) for i in middle_result if not i in stop_word_eng] # 불용어 제거
len(final_result)

#%%
###텍스트에서 많이 나온 단어###
import pandas as pd

english = pd.Series(final_result).value_counts()
print("English top 10")
english

#%%
##CSV 출력##
english.to_csv('key_freq_final.csv')

## 출처: https://wonhwa.tistory.com/23
