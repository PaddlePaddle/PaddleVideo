import json
import math

with open('/workspace/bianjiang03/DATA/Output_for_bmn/prop.json') as f:
    data = json.load(f)
f.close()

transferred = dict()

# 25 fps for all videos
fps = 25

for item in data:
    temp = []
    for seg in item['bmn_results']:
        temp_dict = {
            'score': seg['score'],
            'segment':
            [round(seg['start'] / fps, 2),
             round(seg['end'] / fps, 2)]
        }
        temp.append(temp_dict)
    transferred[item['video_name']] = temp

target_format = {
    'version': 'A-test',
    'results': transferred,
    'external_data': {}
}

jsonString = json.dumps(target_format, indent=4, ensure_ascii=False)
jsonFile = open('/workspace/bianjiang03/DATA/Output_for_bmn/submission.json',
                'w')
jsonFile.write(jsonString)
jsonFile.close()

# target format
# {
#   "version": NA,
#   "results": {
#     "name_of_clip_1": [
#       {
#         "score": 0.64,
#         "segment": [2.33,3.15]
#       },
#       {
#         "score": 0.77,
#         "segment": [7.64, 7.84]
#       }
#     ],
# 	"name_of_clip_2": [
#       {
#         "score": 0.84,
#         "segment": [9.73,10.15]
#       },
#       {
#         "score": 0.87,
#         "segment": [17.11, 17.84]
#       }
#     ],
# 	...
#   }
#   "external_data": {}
# }
