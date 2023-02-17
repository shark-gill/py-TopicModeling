#%%

from wordcloud import WordCloud
from collections import Counter

import pandas as pd

df = pd.read_csv('/home/lhshrk/py-TopicModeling/data/key_freq_rawdata.csv', encoding='utf-8')

words = df.set_index("title").to_dict()["count"]

counts = Counter(words)
tags = counts.most_common(300)

wc = WordCloud(font_path='/home/lhshrk/py-TopicModeling/data/NanumSquareRoundR.ttf',background_color="white", max_font_size=60)
cloud = wc.generate_from_frequencies(dict(tags))
cloud.to_file('/home/lhshrk/py-TopicModeling/data/cloud.png')
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.axis('off')
plt.imshow(cloud)
plt.show()

plt.savefig('/home/lhshrk/py-TopicModeling/result/wc_result.jpg')

#%%
#참고문헌
# https://foreverhappiness.tistory.com/36
# https://blog.naver.com/PostView.nhn?blogId=vi_football&logNo=221775297963&parentCategoryNo=&categoryNo=1&viewDate=&isShowPopularPosts=true&from=search