import os
from itertools import chain
from glob import glob

directory = 'C:/Users/User/Documents/key_network'

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        f = open(filename, 'r')
        text = f.read()
        
        lines = [text.lower() for line in filename]
        with open(filename, 'w') as out:
            out.writelines(lines)


# http://carrefax.com/new-blog/2017/1/16/covert-text-files-to-lower-case-in-python