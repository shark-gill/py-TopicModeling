#%%
from pydoc import doc
from matplotlib.cbook import normalize_kwargs
import pandas as pd
import numpy as np

rawdata = pd.read_csv('test_data2.csv')
documents = pd.DataFrame(rawdata)

len(documents)


#%%
import re
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import preprocess_string

# nltk.download('stopwords')

def clean_text(d):
    pattern = r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', d)
    return text

def clean_stopword(d):
    stop_words = stopwords.words('english')
    return ' '.join([w.lower() for w in d.split() if w.lower() not in stop_words and len(w) > 3])

def preprocessing(d):
    return preprocess_string(d)

# %%
documents.replace("", float("NaN"), inplace=True)
# documents.isnull().values.any() Null 값 확인 코드
documents.dropna(inplace=True)
len(documents)


# %%
documents['abstract'] = documents['abstract'].apply(clean_stopword)

# %%
tokenized_docs = documents['abstract'].apply(preprocessing)
tokenized_docs = tokenized_docs.to_list()

# %%
import numpy as np
drop_docs = [index for index, sentence in enumerate(tokenized_docs) if len(sentence) <=1]
docs_texts = np.delete(tokenized_docs, drop_docs, axis = 0)
len(docs_texts)

# %%
from gensim import corpora

dictionary = corpora.Dictionary(docs_texts)
corpus = [dictionary.doc2bow(text) for text in docs_texts]

# %%
from gensim.models import LsiModel

lsi_model = LsiModel(corpus, num_topics = 10, id2word=dictionary)
topics = lsi_model.print_topics()
topics

# %%
from gensim.models.coherencemodel import CoherenceModel


min_topics, max_topics = 5, 15
coherence_scores = []

for num_topics in range(min_topics, max_topics):
    model = LsiModel(corpus, num_topics=num_topics, id2word=dictionary)
    coherence = CoherenceModel(model=model, texts=docs_texts, dictionary=dictionary)
    coherence_scores.append(coherence.get_coherence())
    
coherence_scores
# %%
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')

x = [int(i) for i in range(min_topics, max_topics)]

plt.figure(figsize=(10, 6))
plt.plot(x, coherence_scores)
plt.xlabel('number of topics')
plt.ylabel('coherence_scores')
plt.show()