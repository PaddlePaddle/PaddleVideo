import copy
import json
import re
import os

url = '/home/aistudio/work/BMN/Input_for_bmn/feature/'
directory = os.fsencode(url)
count = 0
target_set = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    target_name = filename.split('.npy')[0]
    target_set.append(target_name)
    count += 1
print('Feature size:', len(target_set))

with open('/home/aistudio/work/BMN/Input_for_bmn/label.json') as f:
    data = json.load(f)

delet_set = []
for key in data.keys():
    if not key in target_set:
        delet_set.append(key)

print('(Label) Original size:', len(data))
print('(Label) Deleted size:', len(delet_set))

for item in delet_set:
    data.pop(item, None)

print('(Label) Fixed size:', len(data))

jsonString = json.dumps(data, indent=4, ensure_ascii=False)
jsonFile = open('/home/aistudio/work/BMN/Input_for_bmn/label_fixed.json', 'w')
jsonFile.write(jsonString)
jsonFile.close()
