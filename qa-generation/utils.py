import pandas as pd
from datetime import datetime


def load_data(file_path):
    df = pd.read_csv(file_path)
    # df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    # df.drop_duplicates(subset=['title'], keep='first', inplace=True)
    df = df[df['text'].apply(len) > 800]
    df = df[df['text'].apply(len) < 10000]
    df = df[df['title'].str.contains('Opinion:') == False]

    df = df.reset_index(drop = True)
    df['news_id'] = range(len(df))
    return df

def get_prompt(df, prompt_file):
    PROMPTS_ROOT = './qa-generation/prompts/'
    with open(PROMPTS_ROOT + prompt_file) as f:
        prompt_template = f.read()

    dates = df['date']
    texts = df['text']

    if "summary" in prompt_file:
        raw_prompts = [prompt_template.format(texts[i], datetime.strptime(dates[i], "%Y-%m-%d").strftime('%Y-%m-%d, %A')) for i in range(len(df))] 
    elif "gen_qa" in prompt_file:
        key_points = df['keypoint']
        raw_prompts = [prompt_template.format(dates[i],
                                              dates[i],
                                              dates[i],
                                              texts[i],
                                              dates[i],
                                              key_points[i],
                                              datetime.strptime(dates[i], "%Y-%m-%d").strftime("%B %Y"),
                                              datetime.strptime(dates[i], "%Y-%m-%d").strftime("%B %Y"),
                                              datetime.strptime(dates[i], "%Y-%m-%d").strftime("%B %Y"),
                                              dates[i],
                                              dates[i]
                                              )
                                              for i in range(len(df))]
    elif "mcq" in prompt_file:
        qas = df['qa']
        raw_prompts = [prompt_template.format(texts[i],
                                              qas[i][3]['question'],
                                              qas[i][3]['answer'],
                                              qas[i][4]['question'],
                                              qas[i][4]['answer'],
                                              )
                                              for i in range(len(df))]
        
    elif "qa_filter" in prompt_file:
        qas = df['qa']
        raw_prompts = [prompt_template.format(texts[i],
                                              dates[i],
                                              qas[i][1]['question'],
                                              qas[i][1]['answer'],
                                              qas[i][2]['question'],
                                              qas[i][2]['answer'],
                                              qas[i][3]['question'],
                                              qas[i][3]['answer'],
                                              qas[i][4]['question'],
                                              qas[i][4]['answer'],
                                              dates[i],
                                              dates[i],
                                              dates[i],
                                              dates[i],
                                              dates[i]
                                              )
                                              for i in range(len(df))]


    prompts = [format_prompt(p) for p in raw_prompts]
    return prompts

def format_prompt(prompt):
    formatted_prompt =[
        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
        {"role": "user", "content": prompt}
        ]
    return formatted_prompt

