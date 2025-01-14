import os
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import argparse


"""
This script is used in constraint open-book setting, to get the retrived articles from BM25.

RAG_cutoff: the latest accessible date for retrieving articles
The accessible date range for a question with resolution date d_res is 
    [start_date_of_news_corpus, min(d_res − 1, RAG_cutoff))

1) before RAG_cutoff, we have dynamic_cutoff with the retriver having information up to d_res-1,
2) after RAG_cutoff, we have a fixed retreiver with information up to RAG_cutoff
"""

# Constants
NUM_OF_RETRIEVED_ARTICLES = 10

def process_data(data, news, output_file, cutoff_date, dynamic_cutoff):
    """
    Processes the data, retrieves the top k relevant documents from the news dataset,
    and saves the output with the retrieved documents.
    """
    if 'retrieved_news' not in data.columns:
        data['retrieved_news'] = None

    if dynamic_cutoff: # senario 1
        old_yesterday = None
        for idx, row in tqdm(data.iterrows(), total=len(data), desc = 'Retrieving articles under dynamic_cutoff...'):
            # define the cutoff date (1 day before the current row's date)
            yesterday = row['date'] - pd.DateOffset(days=1)
            if yesterday != old_yesterday: # if two rows share the same yesterday, we don't need to define bm25 again
                relevant_df = news[news['date'] < yesterday]
                tokenized_corpus = relevant_df.tokenized_text.to_list()
                bm25 = BM25Okapi(tokenized_corpus)

            tokenized_query = row['tokenized_query']
            doc_scores = bm25.get_scores(tokenized_query)
            # get top k articles
            idx_ls = doc_scores.argsort()[-NUM_OF_RETRIEVED_ARTICLES:][::-1]
            retrieved_df = relevant_df.iloc[idx_ls, :].copy()
            retrieved_df['date'] = retrieved_df['date'].dt.strftime('%Y-%m-%d')
            retrieved_news = retrieved_df[['url', 'source_domain', 'date', 'title', 'text']].to_dict(orient='list')
            old_yesterday = yesterday
            # store the dataframe
            data.at[idx, 'retrieved_news'] = retrieved_news

    else: # senario 2
        cutoff_date = pd.to_datetime(cutoff_date)
        relevant_df = news[news['date'] < cutoff_date]
        tokenized_corpus = relevant_df.tokenized_text.to_list()
        bm25 = BM25Okapi(tokenized_corpus)

        for idx, row in tqdm(data.iterrows(), total=len(data), desc = 'Retrieving articles under fixed RAG_cutoff...'):
            tokenized_query = row['tokenized_query']
            doc_scores = bm25.get_scores(tokenized_query)
            # get top k articles
            idx_ls = doc_scores.argsort()[-NUM_OF_RETRIEVED_ARTICLES:][::-1]
            retrieved_df = relevant_df.iloc[idx_ls, :].copy()
            retrieved_df['date'] = retrieved_df['date'].dt.strftime('%Y-%m-%d')
            retrieved_news = retrieved_df[['url', 'source_domain', 'date', 'title', 'text']].to_dict(orient='list')
            # store the dataframe
            data.at[idx, 'retrieved_news'] = retrieved_news

    # save file
    data.to_pickle(output_file)
    print(f"Data processed and saved to {output_file}")

def main(args):
    cutoff_date = args.cutoff_date
    output_path = args.output_path
    dynamic_cutoff = args.dynamic_cutoff

    output_path = os.path.join(output_path, 'cutoff_'+cutoff_date)
    os.makedirs(output_path, exist_ok=True)

    file_name = os.path.basename(args.question_path).split('.')[0]
    file_name = file_name.replace('_tokenized', '')
    postfix = 'before' if dynamic_cutoff else 'after'
    output_file = os.path.join(output_path, f'rag_{postfix}_{file_name}.pkl')

    # load question data
    print("Loading questions and news...")
    data = pd.read_pickle(args.question_path)
    data['date'] = pd.to_datetime(data['date'])

    if 'tokenized_query' not in data.columns:
        raise ValueError("Missing 'tokenized_query' column in the questions data. Please run rag_tokenize_data.py first.")

    if dynamic_cutoff:
        data = data[data['date'] <= cutoff_date]
    else:
        data = data[data['date'] > cutoff_date]

    if data.empty:
        raise ValueError(f"No data available {postfix} the cutoff date {cutoff_date}.")

    # load news data
    news = pd.read_pickle(args.news_path)
    news['date'] = pd.to_datetime(news['date'])

    if 'tokenized_text' not in news.columns:
        raise ValueError("Missing 'tokenized_text' column in the news data. Please run rag_tokenize_data.py first.")

    process_data(data, news, output_file, cutoff_date, dynamic_cutoff)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BM25 Retriever")
    parser.add_argument('--news_path', type=str, required=True, help='Path to the input news data')
    parser.add_argument('--question_path', type=str, required=True, help='Path to the input question data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output data')
    parser.add_argument('--cutoff_date', type=str, required=True, help='Static cutoff date (yyyy-mm-dd)')
    parser.add_argument('--dynamic_cutoff', action='store_true',
                        help='dynamic_cutoff = True if the question resolution date is before the cutoff date, otherwise = False')

    args = parser.parse_args()
    main(args)