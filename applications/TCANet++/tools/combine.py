import os
import json
import glob

json_path = "/home/aistudio/data/Features_competition_test_B/npy"
json_files = glob.glob(os.path.join(json_path, '*_*.json'))

submit_dic = {"version": None,
              "results": {},
              "external_data": {}
              }
results = submit_dic['results']
for json_file in json_files:
    j = json.load(open(json_file, 'r'))
    old_video_name = list(j.keys())[0]
    video_name = list(j.keys())[0].split('/')[-1].split('.')[0]
    video_name, video_no = video_name.split('_')
    start_id = int(video_no) * 12
    if len(j[old_video_name]) == 0:
        continue
    for i, top in enumerate(j[old_video_name]):
        if video_name in results.keys():
            results[video_name].append({'score': round(top['score'], 2),
                                        'segment': [round(top['segment'][0] + start_id, 2), round(top['segment'][1] + start_id, 2)]})
        else:
            results[video_name] = [{'score':round(top['score'], 2),
                                        'segment': [round(top['segment'][0] + start_id, 2), round(top['segment'][1] + start_id, 2)]}]

json.dump(submit_dic, open('/home/aistudio/submission1.json', 'w', encoding='utf-8'))