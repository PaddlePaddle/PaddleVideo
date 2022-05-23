import copy
import json
import re
import os
from sklearn.model_selection import KFold

folds = 5
fold = 1
kf = KFold(n_splits=folds,shuffle=True,random_state=42)

with open('/home/aistudio/data/new_label_cls14_train.json') as f:
    data = json.load(f)

data_list = []
for key in data.keys():
    data_list.append(key)

for train_index , test_index in kf.split(data_list):
    fold_delet_set = []
    Data = data.copy()
    for index in test_index:
        fold_delet_set.append(data_list[index])
    for item in fold_delet_set:
        Data.pop(item, None)
    print(f"fold{fold}:{len(Data)}/{len(data)}")
    jsonString = json.dumps(Data, indent=4, ensure_ascii=False)
    jsonFile = open(f"/home/aistudio/data/new_label_cls14_train_fold_{fold}.json", 'w')
    jsonFile.write(jsonString)
    jsonFile.close()
    fold = fold + 1