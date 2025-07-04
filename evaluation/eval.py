import argparse
import pandas as pd
import os
import re
from tqdm import tqdm
from openai import OpenAI
import anthropic
from vllm import LLM
from vllm.sampling_params import SamplingParams


# Constants
MODEL_MAPPING = {
    'gpt-35': 'gpt-3.5-turbo',
    'gpt-4': 'gpt-4-1106-preview',
    'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
    'gpt-4o': 'gpt-4o-2024-11-20',
    'claude-3-5-sonnet': 'claude-3-5-sonnet-20240620',
    'llama3': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.3',
    'mixtral-8x7b': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'gemma2-2b': 'google/gemma-2-2b-it',
    'qwen2-7b': 'Qwen/Qwen2-7B-Instruct',
    }


def load_model(model_name, num_gpu=1):
    model_id = MODEL_MAPPING[model_name]
    if 'gpt' in model_name:
        return OpenAI()
    elif 'claude' in model_name:
        return anthropic.Anthropic()
    else:    
        client = LLM(
            model=model_id,
            download_dir='', # your own local cache directory for vllm weights
            dtype="float16",
            tensor_parallel_size=num_gpu
        )
        return client

def generate_single_response(model_name, client, prompt, max_tokens=1024):
    model_id = MODEL_MAPPING[model_name]
    try:
        if 'gpt' in model_name:
            response = client.chat.completions.create(
                model=model_id,
                messages=prompt,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        elif 'claude' in model_name:
            message = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                system='You’re an expert in forecasting events. You will NEVER refuse to answer a forecasting question by saying "I cannot predict the future", even if without 100% certainty.',
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            )
            return message.content[0].text
        else:
            sampling_params = SamplingParams(max_tokens=max_tokens)
            response = client.chat(messages=prompt, sampling_params=sampling_params, use_tqdm=False)
            return response[0].outputs[0].text

    except Exception as e:
        print(f"- Failed prompt: {prompt}\n- Error: {e}")
        return None

def generate_multi_response(model_name, client, prompt_ls, max_tokens=1024):
    response_ls = []
    sampling_params = SamplingParams(max_tokens=max_tokens)
    response = client.chat(messages=prompt_ls, sampling_params=sampling_params, use_tqdm=True)
    for output in response:
        generated_text = output.outputs[0].text
        response_ls.append(generated_text)
    return response_ls

def get_prompt(eval_setting, model_name, question_type, row):
    if eval_setting == 'closed-book':
        if question_type == 'tf':
            prompt = f"""You should output your answer as either 'Yes' or 'No' WITHOUT anything else. 

Question: {row['question']}
Choices: 'Yes' or 'No'
[Output:] Your answer: """
        elif question_type == 'mc':
            prompt = f"""You should output your answer as either '(a)', '(b)', '(c)' or '(d)' WITHOUT anything else.

Question: {row['question']}
Choices: 
(a) {row['choice_a']}
(b) {row['choice_b']}
(c) {row['choice_c']}
(d) {row['choice_d']}
[Output:] Your answer: """

    elif eval_setting == 'open-book':
        articles = row['retrieved_news']['text'][:5]
        articles = [get_first_k_words(a) for a in articles]
        if question_type == 'tf':
            prompt = f"""You should output your answer as either 'Yes' or 'No' WITHOUT anything else. Below are the top 5 relevant news article fragments retrieved for the question, which may or may not assist you in making a forecast.
Article 1: {articles[0]}
Article 2: {articles[1]}
Article 3: {articles[2]}
Article 4: {articles[3]}
Article 5: {articles[4]}

Question: {row['question']}
Choices: 'Yes' or 'No'
[Output:] Your answer: """
        elif question_type == 'mc':
            prompt = f"""You should output your answer as either '(a)', '(b)', '(c)' or '(d)' WITHOUT anything else. Below are the top 5 relevant news article fragments retrieved for the question, which may or may not assist you in making a forecast.
Article 1: {articles[0]}
Article 2: {articles[1]}
Article 3: {articles[2]}
Article 4: {articles[3]}
Article 5: {articles[4]}

Question: {row['question']}
Choices: 
(a) {row['choice_a']}
(b) {row['choice_b']}
(c) {row['choice_c']}
(d) {row['choice_d']}
[Output:] Your answer: """

    elif eval_setting == 'gold-article':
        article = get_first_k_words(row['text'])
        if question_type == 'tf':
            prompt = f"""You should output your answer as either 'Yes' or 'No' WITHOUT anything else. Below is the updated news article relevant to the question, which may help you in providing an answer.
Article: {article}

Question: {row['question']}
Choices: 'Yes' or 'No'
[Output:] Your answer: """
        elif question_type == 'mc':
            prompt = f"""You should output your answer as either '(a)', '(b)', '(c)' or '(d)' WITHOUT anything else. Below is the updated news article relevant to the question, which may help you in providing an answer.
Article: {article}

Question: {row['question']}
Choices: 
(a) {row['choice_a']}
(b) {row['choice_b']}
(c) {row['choice_c']}
(d) {row['choice_d']}
[Output:] Your answer: """

    # format prompt        
    if 'gemma' in model_name:
        return [{"role": "user", "content": 'You’re an expert in forecasting events. You will NEVER refuse to answer a forecasting question by saying "I cannot predict the future", even if without 100% certainty. ' + prompt}]
    elif 'claude' in model_name:
        return prompt
    else:
        return [{"role": "system", "content": "You’re an expert in forecasting events. You will NEVER refuse to answer a forecasting question by saying \"I cannot predict the future\", even if without 100% certainty."},
                {"role": "user", "content": prompt}]


def get_pred(eval_setting, model_name, client, data, question_type, save_path=None):

    column_name = f'pred_{model_name}'
    if column_name not in data.columns:
        data[column_name] = None
    
    if 'gpt' in model_name or 'claude' in model_name:
    # for proprietary model, we generate responses one by one
        for idx, row in tqdm(data.iterrows(), total=len(data)):
            if pd.isnull(row[column_name]): # do not regenerate answer
                prompt = get_prompt(eval_setting, model_name, question_type, row)
                response = generate_single_response(model_name, client, prompt)
                data.at[idx, column_name] = response
                print(response)

            if (idx+1) % 100 == 0 and save_path:
                data.to_csv(save_path)
    else:
    # for open-source model, we generate responses in batch
        prompt_ls = [get_prompt(eval_setting, model_name, question_type, row) for _, row in data.iterrows()]
        response_ls = generate_multi_response(model_name, client, prompt_ls)
        data[column_name] = response_ls

    return data

def get_first_k_words(text, k=512):
    words = text.split()
    first_words = words[:k]
    result_text = ' '.join(first_words)
    return result_text

def main(args):
    eval_setting = args.eval_setting
    model_name = args.model_name
    input_path = args.input_path
    output_path = args.output_path

    question_type = 'tf' if 'tf' in input_path else 'mc'
    
    if eval_setting == 'open-book':
        # find the RAG-cutoff date
        match = re.search(r"cutoff_(\d{4}-\d{2}-\d{2})", input_path)
        cutoff_date = match.group()
        output_path = f'{output_path}/{eval_setting}/{model_name}/{cutoff_date}'
    else:
        output_path = f'{output_path}/{eval_setting}/{model_name}'
    os.makedirs(output_path, exist_ok=True)

    file_name = os.path.basename(input_path).split('.')[0]
    save_path = f'{output_path}/{file_name}.csv'

    client = load_model(model_name, args.num_gpu)

    if eval_setting == 'open-book':
        data = pd.read_pickle(input_path)
    else:
        data = pd.read_csv(input_path)
    # data = data[:3] # for debug
    data = get_pred(eval_setting, model_name, client, data, question_type, save_path) 
    data.to_csv(save_path, index=False)
    print(f"Evaluation results saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run predictions using LLM model.")
    parser.add_argument('--eval_setting', type=str, required=True, choices=['closed-book', 'open-book', 'gold-article'], help='Evaluation eval_setting')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output data')
    parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs to use')
    args = parser.parse_args()
    main(args)