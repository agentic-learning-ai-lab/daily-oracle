import argparse
import pandas as pd
import re
import os
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

def preprocess_text(text):
    """Clean and preprocess text data."""
    tokens = nltk.word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z]', '', token.lower()) for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def tokenize_news(news, save_path):
    """Tokenize the news text and save it."""
    tokenized_corpus = [preprocess_text(doc) for doc in tqdm(news['text'], desc="Tokenizing news")]
    news.drop(columns=['Unnamed: 0'], inplace=True)
    news['tokenized_text'] = tokenized_corpus
    news.to_pickle(save_path)

def tokenize_questions(data, save_path):
    """Tokenize the question and save it."""
    tokenized_query = [preprocess_text(q) for q in tqdm(data['question'], desc="Tokenizing questions")]
    data['tokenized_query'] = tokenized_query
    data.to_pickle(save_path)

def main(args):
    news_path = args.news_path
    question_path = args.question_path
    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)

    if news_path:
        news_file_name = os.path.basename(news_path).split('.')[0]
        news_save_path = os.path.join(output_dir, f'{news_file_name}_tokenized.pkl')
        news = pd.read_csv(news_path)
        tokenize_news(news, news_save_path)

    if question_path:
        question_file_name = os.path.basename(question_path).split('.')[0]
        question_save_path = os.path.join(output_dir, f'{question_file_name}_tokenized.pkl')
        data = pd.read_csv(question_path)
        tokenize_questions(data, question_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tokenize data for RAG setting")
    parser.add_argument('--news_path', type=str, required=False, help='Path to the input news data')
    parser.add_argument('--question_path', type=str, required=False, help='Path to the input question data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output data')
    args = parser.parse_args()

    if not args.news_path and not args.question_path:
        raise ValueError("At least one of --news_path or --question_path must be provided.")

    main(args)