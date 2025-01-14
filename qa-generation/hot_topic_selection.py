"""
Pipeline: News Article Preprocessing - Vectorization - Clustering - Article Selection
"""

import argparse
import os
import random
import re
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
from collections import Counter

from utils import load_data


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Constants
DEFAULT_NUM_ARTICLES = 3

def preprocess_text(text):
    """Clean and preprocess text data."""
    tokens = nltk.word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z]', '', token.lower()) for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)


def vectorize_text(news_text, n_components=100):
    """
    TF-IDF vectorization and dimension reducation of text data.
    """
    print("Preprocessing text data...")
    preprocessed_text = [preprocess_text(text) for text in news_text]
    
    print("Vectorizing text data with TF-IDF...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_text)
    
    if n_components and len(news_text) >= n_components:
        print("Reducing dimensionality with PCA...")
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X.toarray())
    
    return X


def adjust_dbscan_params(X, k=5):
    """
    Adjust DBSCAN parameters `eps` and `min_samples` based on the dataset X.
    """
    if X.shape[0] < k:
        return 0.5, 2  # Default fallback for small datasets
    
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    sorted_distances = np.sort(distances[:, k - 1], axis=0)
    
    eps = np.percentile(sorted_distances, 90)
    min_samples = max(2, int(np.log(len(X))))
    
    print(f"Adjusted DBSCAN Params → eps: {eps:.4f}, min_samples: {min_samples}")
    return eps, min_samples


def cluster_texts(X, eps=0.5, min_samples=3):
    """
    Cluster vectorized news articles using DBSCAN.
    """
    print("Clustering text data using DBSCAN...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    dbscan.fit(X)
    return dbscan.labels_


def select_top_articles(data, labels, X, avg_distance_threshold=0.6):
    """
    Select top articles - one from each top-k valid clusters with largest cluster size.
    Parameters:
        - data (pd.DataFrame): The article dataset.
        - labels (array): Cluster labels from DBSCAN.
        - X (array): Cluster data points.
        - avg_distance_threshold (float): Max average pairwise distance for valid clusters.
    Returns:
        - selected_articles (pd.DataFrame): Selected top articles.
    """
    # Group indices by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    valid_clusters = []
    
    # Check for validity of each cluster based on average pairwise distance
    for cluster_id, indices in clusters.items():
        if cluster_id == -1 or len(indices) < 2:
            continue  # Skip noise and tiny clusters
        
        cluster_points = X[indices]
        avg_distance = np.mean(pairwise_distances(cluster_points, metric='cosine'))
        
        # Only consider clusters with avg_distance <= avg_distance_threshold
        if avg_distance <= avg_distance_threshold:
            valid_clusters.append((cluster_id, len(indices), avg_distance))
    
    # Sort valid clusters by size (descending)
    valid_clusters = sorted(valid_clusters, key=lambda x: x[1], reverse=True)
    
    # Sanity check
    print(f"--- Titles in Top-{DEFAULT_NUM_ARTICLES} Valid Clusters ---")
    for i, (cluster_id, size, avg_distance) in enumerate(valid_clusters[:DEFAULT_NUM_ARTICLES], start=1):
        cluster_indices = [idx for idx, lbl in enumerate(labels) if lbl == cluster_id]
        cluster_titles = data.iloc[cluster_indices]['title'].tolist()
        print(f"\nCluster {i} (ID: {cluster_id}, Size: {size}, Avg Distance: {avg_distance:.4f}) Titles:")
        for title in cluster_titles:
            print(f"- {title}")
    print("--------------------------------------\n")
    
    selected_indices = set()
    selected_articles = []
    
    # Pick one random article from each of the top-K valid clusters
    for cluster_id, _, _ in valid_clusters[:DEFAULT_NUM_ARTICLES]:
        idx = random.choice(clusters[cluster_id])
        selected_indices.add(idx)
        selected_articles.append(data.iloc[[idx]])
    
    # Add random articles if fewer than DEFAULT_NUM_ARTICLES are selected
    while len(selected_articles) < DEFAULT_NUM_ARTICLES:
        idx = random.randint(0, len(data) - 1)
        if idx not in selected_indices:
            selected_indices.add(idx)
            selected_articles.append(data.iloc[[idx]])
    
    return pd.concat(selected_articles, ignore_index=True)


def process_date(date, data, output_path):
    """
    Process and save selected articles for a specific date.
    """
    daily_data = data[data['date'] == date]
    if daily_data.empty:
        print(f"No data for date: {date}")
        return
    
    news_text = daily_data['text'].tolist()
    X = vectorize_text(news_text)
    # eps, min_samples = adjust_dbscan_params(X)
    eps, min_samples = 0.5, 3
    labels = cluster_texts(X, eps=eps, min_samples=min_samples)
    
    # Sanity check - cluster statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_points = list(labels).count(-1)
    n_grouped_points = len(labels) - n_noise_points
    total_samples = len(labels)
    
    print(f"\n--- Cluster Statistics ---")
    print(f"Number of clusters: {n_clusters}")
    print(f"Total number of samples: {total_samples}")
    print(f"Number of grouped points: {n_grouped_points}")
    print(f"Number of noise points: {n_noise_points}")
    print("---------------------------\n")
    
    selected_articles = select_top_articles(
        daily_data,
        labels,
        X,
        avg_distance_threshold=0.7
    )
    
    save_path = os.path.join(output_path, date)
    os.makedirs(save_path, exist_ok=True)
    selected_articles.to_csv(os.path.join(save_path, 'articles_selected.csv'), index=False)
    print(f"Articles for {date} saved successfully!")


def main(args):
    data = load_data(args.input_path)
    
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    date_range = pd.date_range(start=start_date, end=end_date)
    
    for current_date in tqdm(date_range, desc="Processing Dates"):
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"\nProcessing date: {date_str}")
        process_date(date_str, data, args.output_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hot Topic Selection Pipeline")
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output data')
    
    args = parser.parse_args()
    main(args)