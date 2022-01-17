import json

with open('/home/aistudio/data/label_cls14_train.json') as f:
    data = json.load(f)
f.close()

val = {'gts': data['gts'][0:5], 'fps': 25}

jsonString = json.dumps(val, indent=4, ensure_ascii=False)
jsonFile = open('/home/aistudio/data/label_cls14_val.json', 'w')
jsonFile.write(jsonString)
jsonFile.close()

train = {'gts': data['gts'][5:], 'fps': 25}

jsonString = json.dumps(train, indent=4, ensure_ascii=False)
jsonFile = open('/home/aistudio/data/label_cls14_train.json', 'w')
jsonFile.write(jsonString)
jsonFile.close()
