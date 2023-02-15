import csv
import json

csvfile = open('rawdata.csv', 'r', encoding='utf-8')
jsonfile = open('rawdata.json', 'w')

fieldnams = ("searchWord", "Publication Year", "Authors", "journal", "Article Title", "Kewords", "abstract")
reader = csv.DictReader(csvfile, fieldnams)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')
    

# 참고 url
# https://shelling203.tistory.com/42