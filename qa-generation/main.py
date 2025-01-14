import argparse
import os
import re
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from datetime import datetime, timedelta
from utils import load_data, get_prompt


# Constants
MODEL_MAPPING = {
    'gpt-35': 'gpt-3.5-turbo',
    'gpt-4': 'gpt-4-1106-preview',
    'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
    'gpt-4o': 'gpt-4o-2024-11-20'
}
DEFAULT_NUM_ARTICLES = 3 # number of articles to generate QA pairs for each day
SUMMARY_MODEL = 'gpt-4o-mini' #'gpt-4'
QA_MODEL = 'gpt-4o' #'gpt-4'
MCQ_MODEL = 'gpt-4o'#'gpt-4'
FILTER_MODEL = 'gpt-4o-mini' #'gpt-35'

def generate_response(client, prompt, model_name='gpt-4', max_new_tokens=1024):
    """Call OpenAI API."""
    try:
        model_ = MODEL_MAPPING.get(model_name, 'gpt-4-1106-preview')
        response = client.chat.completions.create(
            model=model_,
            messages=prompt,
            max_tokens=max_new_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"- Failed prompt: {prompt}\n- Error: {e}")
        return None


def parse_summary_response(response):
    """Parse the summary and keypoint from the LLM response."""
    if not response:
        return None, None

    split_keys = ["Keypoint:", "Key Point:", "Key point:"]
    for key in split_keys:
        if key in response:
            split_text = response.split(key)
            summary = split_text[0].replace("Summary:", "").strip()
            key_point = split_text[1].strip()
            return summary, key_point
    return None, None


def generate_summary(client, df):
    """Generate article summaries."""
    prompts = get_prompt(df, "summary.prompt")
    summaries, key_points = [], []

    for prompt in tqdm(prompts, desc="Generating summaries"):
        res = generate_response(client, prompt, model_name=SUMMARY_MODEL)
        summary, key_point = parse_summary_response(res)
        summaries.append(summary)
        key_points.append(key_point)

    df['summary'] = summaries
    df['keypoint'] = key_points
    return df


def parse_qa_response(response):
    """Parse questions and answers from the LLM response."""
    qa_dict = {}
    if not response:
        return qa_dict

    try:
        parts = response.split("Question ")
        for part in parts[1:]:
            question_split = part.split('Answer')
            question_number = int(question_split[0].split(":")[0].strip())
            question_text = question_split[0].split(":")[1].strip()
            answer_text = question_split[1].split(":")[1].strip()
            qa_dict[question_number] = {'question': question_text, 'answer': answer_text}
    except Exception as e:
        print(f"- QA Parsing Error: {e}")
    return qa_dict


def generate_qa(client, df):
    """Generate QA pairs."""
    prompts = get_prompt(df, "gen_qa.prompt")
    qa_list = []

    for prompt in tqdm(prompts, desc="Generating QA pairs"):
        res = generate_response(client, prompt, model_name=QA_MODEL)
        qa_list.append(parse_qa_response(res))

    df['qa'] = qa_list
    return df


def parse_mcq_response(response):
    """Parse misleading choices for MCQs."""
    try:
        return re.findall(r'\([bcd]\) ([^\n]+)', response)
    except Exception as e:
        print(f"- MCQ Parsing Error: {e}")
        return []


def generate_mcq(client, df):
    """Generate misleading choices for MCQs."""
    prompts = get_prompt(df, "mcq.prompt")

    for i, prompt in enumerate(tqdm(prompts, desc="Generating misleading choices for MCQs")):
        res = generate_response(client, prompt, model_name=MCQ_MODEL)
        choices = parse_mcq_response(res)
        if len(choices) >= 6:
            df['qa'][i][3]['choices'] = choices[:3]
            df['qa'][i][4]['choices'] = choices[3:]

    return df


def qa_filter(client, df):
    """Generate QA filter results."""
    prompts = get_prompt(df, "qa_filter.prompt")
    responses = []

    for prompt in tqdm(prompts, desc="Filtering QA"):
        res = generate_response(client, prompt, model_name=FILTER_MODEL, max_new_tokens=2048)
        responses.append(res)

    df['qa_filter'] = responses
    return df


def process_date(client, article_selection, date, df, output_path):
    """Process data for a single date."""
    if article_selection == 'random':
        daily_data = df[df['date'] == date]
        if len(daily_data) > DEFAULT_NUM_ARTICLES:
            daily_data = daily_data.sample(n=DEFAULT_NUM_ARTICLES, random_state=42).reset_index(drop=True)
    elif article_selection == 'selected':
        daily_data = df

    # create a directory for each date
    save_path = os.path.join(output_path, date)
    os.makedirs(save_path, exist_ok=True)

    # Step 1: Article Summary
    daily_data = generate_summary(client, daily_data)
    daily_data = daily_data[daily_data['keypoint'].notnull()].reset_index(drop=True)

    # Step 2: QA Generation
    daily_data = generate_qa(client, daily_data)
    daily_data = daily_data[daily_data['qa'].apply(len) == 4].reset_index(drop=True)

    # Step 3: Misleading Choices Generation
    daily_data = generate_mcq(client, daily_data)

    # Step 4: QA Filtering
    daily_data = qa_filter(client, daily_data)

    # save file
    final_file = os.path.join(save_path, f'qa_{article_selection}.pkl')
    daily_data.to_pickle(final_file)

    print(f"Data generation completed for {date}, and file is saved to: {final_file}")


def main(args):
    client = OpenAI()

    article_selection = args.article_selection

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    input_path = args.input_path
    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)
    date_range = pd.date_range(start=start_date, end=end_date)

    if article_selection == 'random':
        news_df = load_data(input_path)
        for current_date in tqdm(date_range, desc="Processing Dates"):
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"\nProcessing date: {date_str}")
            process_date(client, article_selection, date_str, news_df, output_dir)

    elif article_selection == 'selected':
        for current_date in tqdm(date_range, desc="Processing Dates"):
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"\nProcessing date: {date_str}")
            daily_df = pd.read_csv(f'{input_path}/{date_str}/articles_selected.csv').reset_index(drop=True)
            process_date(client, article_selection, date_str, daily_df, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Daily Oracle QA Generation Pipeline")
    parser.add_argument('--article_selection', type=str, required=True, choices=['random', 'selected'], help='Specify article selection type: "random" or "selected"')
    parser.add_argument('--start_date', type=str, required=True, help='Specify the start date in YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, required=True, help='Specify the end date in YYYY-MM-DD')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output data')
    args = parser.parse_args()
    main(args)