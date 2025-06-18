import json
import random as rd
save_rate = 1.01

f = open('./hellaswag_val.jsonl',encoding='utf-8')
fw = open('hellaswag_sample_val.jsonl','w', encoding='utf-8')
for line in f:
    jsonline = json.loads(line)
    ans = jsonline['endings'][jsonline["label"]]
    jsonline['endings'] = rd.sample(jsonline['endings'], k=len(jsonline['endings']))
    jsonline["label"] = jsonline['endings'].index(ans)

    if rd.random() < save_rate:
        fw.write(json.dumps(jsonline, ensure_ascii=False)+ '\n')
