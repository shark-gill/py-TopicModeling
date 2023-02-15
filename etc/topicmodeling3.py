#%%
import pandas as pd
import numpy as np

rawdata = pd.read_csv('test_data2.csv')
documents = pd.DataFrame(rawdata)

print(len(documents))

# %%
