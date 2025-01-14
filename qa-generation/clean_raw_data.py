import argparse
import pandas as pd
import os
from datetime import datetime
import random
import re


QA_FILTER_THRESHOLD = 13

def get_points(text):
    pattern = r'##\s*(.*?)\s*-\s*(?:Reasoning:.*?)?Points?:\s*(\d+)\s*'
    matches = re.findall(pattern, text, re.DOTALL)
    
    points = {}
    if matches:
        for category, score in matches:
            try:
                points[category.strip()] = int(score)
            except ValueError:
                points[category.strip()] = None

        # Calculate total points
        total = sum(value for value in points.values() if isinstance(value, int))
        points['total'] = total

    return points

def process_qa_filter(data):
    for index, row in data.iterrows():
        text = row.qa_filter
        if text is not None:
            parts = text.split("*Question ")
            
            for part in parts[1:]:  # skip the first empty element
                points = get_points(part)
                
                question_num = int(part[0])  # extract question number
                row.qa[question_num]['points'] = points  # assign points to the corresponding question
                row.qa[question_num]['qa_filter'] = part
        else:
            print('Missing QA filter response')
    
    # Check if all points are fully extracted from the qa_filter
        error_count = 0
        try:
            if not all(len(row['qa'][i]['points']) == 8 for i in [1, 2, 3, 4]):
                error_count += 1
        except Exception as e:
            print(f"- Score Check Error: {e}")
            print(f"QA: {row['qa']}")
            print(f"QA Filter: {row['qa_filter']}")

    print(f"- Total rows processed: {len(data)}")
    print(f"- QA filter error count: {error_count}")

    return data

def get_ls(qas, i, colname):
    l = []
    for qa in qas:
        try:
            l.append(qa[i][colname])
        except:
            l.append(None)
    return l

def get_choices(data):
    """Add all choices and shuffle them for MCQ"""
    answer_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

    for i in range(len(data)):
        for idx in [3, 4]:
            try:
                answer = data.qa[i][idx]['answer']
                choices = data.qa[i][idx]['choices']
                all_choices = choices.copy() + [answer]
                random.shuffle(all_choices)
                data.qa[i][idx]['all_choices'] = all_choices
                answer_idx = all_choices.index(answer)
                data.qa[i][idx]['answer'] = answer_map[answer_idx]
            except Exception as e:
                print(f"- Get choices error: {e}")
    return data

def filter_final_data(df):
    df = df[df['total_points'] >= QA_FILTER_THRESHOLD]
    df = df.dropna(subset=['keypoint'])
    df = df[~df['keypoint'].str.contains("No new event", case=False)]
    df = df[df['question'].str.contains(r'\?')]
    df = df[~df['question'].str.contains(r'\baccording\b', case=False)]
    df = df[~df['question'].str.contains(r'\barticle\b', case=False)]
    df['question'] = df['question'].str.strip('"')
    df['question'] = df['question'].str.strip("'")
    # sort df by date (ascending order)
    df['date'] = pd.to_datetime(df['date'])  # Ensure 'date' is in datetime format
    df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean_and_save_data(data, output_path, start_date, end_date):
    # TF Questions
    clean_df = pd.DataFrame()
    for i in [1,2]:
        temp_df = pd.DataFrame()
        temp_df['question'] = [qa[i]['question'] for qa in data.qa]
        temp_df['answer'] = [qa[i]['answer'] for qa in data.qa]
        temp_df['date'] = data.date
        temp_df['article_selection'] = data.article_selection
        temp_df['title'] = data.title
        temp_df['text'] = data.text
        temp_df['summary'] = data.summary
        temp_df['keypoint'] = data.keypoint
        temp_df['url'] = data.url
        temp_df['source_domain'] = data.source_domain
        temp_df['qa_filter'] = get_ls(data.qa, i, 'qa_filter')
        temp_df['points'] = get_ls(data.qa, i, 'points')
        temp_df['total_points'] = [d['total'] if d!=None else None for d in temp_df['points']]
        temp_df.drop(columns=['points'], inplace=True)
        clean_df = pd.concat([clean_df, temp_df])

    clean_df = filter_final_data(clean_df)
    clean_df.to_csv(f'{output_path}/tf_questions_{start_date}_{end_date}.csv', index=False)
    print(f"- {len(clean_df)} TF QA pairs saved to {output_path}/tf_questions_{start_date}_{end_date}.csv")

    # MC Questions
    clean_df = pd.DataFrame()
    temp_df = pd.DataFrame()
    data = get_choices(data)
    for i in [3,4]:
        temp_df['question'] = [qa[i]['question'] for qa in data.qa]
        choices_mcq = get_ls(data.qa, i, 'all_choices')
        choice_a = [choices[0] for choices in choices_mcq]
        choice_b = [choices[1] for choices in choices_mcq]
        choice_c = [choices[2] for choices in choices_mcq]
        choice_d = [choices[3] for choices in choices_mcq]
        temp_df['choice_a'] = choice_a
        temp_df['choice_b'] = choice_b
        temp_df['choice_c'] = choice_c
        temp_df['choice_d'] = choice_d
        temp_df['answer'] = get_ls(data.qa, i, 'answer')
        temp_df['date'] = data.date
        temp_df['article_selection'] = data.article_selection
        temp_df['title'] = data.title
        temp_df['text'] = data.text
        temp_df['summary'] = data.summary
        temp_df['keypoint'] = data.keypoint
        temp_df['url'] = data.url
        temp_df['source_domain'] = data.source_domain
        temp_df['qa_filter'] = get_ls(data.qa, i, 'qa_filter')
        temp_df['points'] = get_ls(data.qa, i, 'points')
        temp_df['total_points'] = [d['total'] if d!=None else None for d in temp_df['points']]
        temp_df.drop(columns=['points'], inplace=True)
        clean_df = pd.concat([clean_df, temp_df])
    
    clean_df = filter_final_data(clean_df)
    clean_df.to_csv(f'{output_path}/mc_questions_{start_date}_{end_date}.csv', index=False)
    print(f"- {len(clean_df)} MC QA pairs saved to {output_path}/mc_questions_{start_date}_{end_date}.csv")

def main(args):
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    date_range = pd.date_range(start=start_date, end=end_date)

    # Combine daily data
    all_dfs = []
    for current_date in date_range:
        date_str = current_date.strftime("%Y-%m-%d")
        try:
            random_df = pd.read_pickle(f"{input_path}/{date_str}/qa_random.pkl")
            random_df['article_selection'] = 'random'
            selected_df = pd.read_pickle(f"{input_path}/{date_str}/qa_selected.pkl")
            selected_df['article_selection'] = 'selected'
            all_dfs.append(pd.concat([random_df, selected_df], ignore_index=True))
        except FileNotFoundError:
            print(f"Data files missing for {date_str}, skipping...")
            continue

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.reset_index(drop=True, inplace=True)
    else:
        print("No data found for the given date range.")

    # Process the responses in the QA filter step and get the points
    final_df = process_qa_filter(final_df) 

    # Clean and save TF and MC QA pairs
    final_df = clean_and_save_data(final_df, output_path, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean raw data to a csv of QA pairs")
    parser.add_argument('--start_date', type=str, required=True, help='Specify the start date in YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, required=True, help='Specify the end date in YYYY-MM-DD')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output data')
    args = parser.parse_args()
    main(args)