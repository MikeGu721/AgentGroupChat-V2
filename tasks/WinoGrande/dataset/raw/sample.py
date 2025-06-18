import json
import random as rd

f = open('./dev.jsonl',encoding='utf-8')
fw = open('dev_sample.jsonl','w', encoding='utf-8')
options = set()
num, leave = 0, 0,
for line in f:
    num += 1
    jsonline = json.loads(line)
    jsonline_options = str(sorted([jsonline['option1'], jsonline['option2']]))
    if jsonline_options in options: continue

    options.add(jsonline_options)
    jsonline['option1'], jsonline['option2'] = jsonline['option2'], jsonline['option1']
    jsonline['answer'] = str(int(jsonline['answer'])%2+1)
    
    fw.write(json.dumps(jsonline, ensure_ascii=False)+ '\n')
    leave += 1
print(num, leave)
