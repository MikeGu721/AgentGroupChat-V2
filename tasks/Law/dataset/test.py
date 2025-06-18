import json


def make_data():
    train_1 = list()
    train_0 = list()
    with open('./fakao_gpt4.json', 'r', encoding='utf-8') as f:
        qa_data = json.loads(f.read())
    with open('./JEC-QA/1_train.json', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            index = json.loads(line.strip())
            train_1.append(index)
    with open('./JEC-QA/0_train.json', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            index = json.loads(line.strip())
            train_0.append(index)
    for qa in train_0:
        for sft in qa_data:
            if qa['statement'] in sft['input']:
                answer = ''
                for a in qa['answer']:
                    answer += a
                sft['answer'] = answer
    for qa in train_1:
        for sft in qa_data:
            if qa['statement'] in sft['input']:
                answer = ''
                for a in qa['answer']:
                    answer += a
                sft['answer'] = answer
    rm_no_select = list()
    for line in qa_data:
        if 'Question' not in line['input']:
            continue
        else:
            rm_no_select.append({
                'problem': line['input'],
                'solution': line['output'],
                'ground_truth': line['answer']
            })
    with open('./AgentGroupChat_LawExam_zh.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(rm_no_select, ensure_ascii=False, indent=1))


from openai import OpenAI


def run_api(prompt, stream=False):
    gpt_params = {
        "temperature": 0.8,
        "top_p": 1,
        "stream": stream,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None
    }
    message = [{
        "role": "user",
        "content": prompt
    }]
    client = OpenAI(
        base_url='https://cn2us02.opapi.win/v1',
        api_key='sk-BOATTlUU478801532161T3BLbkFJA37E314ccd1949efa37d'
    )
    retry = 10
    while True:
        try:
            completion = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=message,
                temperature=gpt_params["temperature"],
                top_p=gpt_params["top_p"],
                stream=gpt_params["stream"],
                frequency_penalty=gpt_params["frequency_penalty"],
                presence_penalty=gpt_params["presence_penalty"],
                stop=gpt_params["stop"]
            )
            result = completion.choices[0].message.content
            break
        except Exception as e:
            if retry > 0:
                retry -= 1
                continue
            raise ConnectionError('Api Failed with Exception {}'.format(e))
    return result


def translate_data():
    with open('./AgentGroupChat_LawExam_zh.json', 'r', encoding='utf-8') as f:
        data_zh = json.loads(f.read())
    data_en = list()
    for line in data_zh:
        template = '''
        把以下内容翻译成地道的英文：
        {chinese}
        '''
        prompt_problem = template.format(chinese=line['problem'])
        prompt_solution = template.format(chinese=line['solution'])
        response_problem = run_api(prompt_problem)
        response_solution = run_api(prompt_solution)
        data_en.append({
            'problem': response_problem,
            'solution': response_solution,
            'ground_truth': line['ground_truth']
        })
    with open('./AgentGroupChat_LawExam_en.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(data_en, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    # make_data()
    translate_data()
