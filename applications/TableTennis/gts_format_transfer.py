import json

with open('/home/aistudio/work/BMN/Input_for_bmn/label_fixed.json') as f:
    data = json.load(f)
f.close()

target_format = {'taxonomy': None, 'database': data, 'version': None}

jsonString = json.dumps(target_format, indent=4, ensure_ascii=False)
jsonFile = open('/home/aistudio/work/BMN/Input_for_bmn/label_gts.json', 'w')
jsonFile.write(jsonString)
jsonFile.close()
