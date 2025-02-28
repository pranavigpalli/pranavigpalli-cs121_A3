import json
import time
import heapq
import math
from math import log10
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download NLTK data
nltk.download('punkt')

# Load the inverted index from the JSON file
with open('inverted_index.json', 'r', encoding='utf-8') as file:
    inverted_index = json.load(file)

# Initialize the stemmer
stemmer = PorterStemmer()

# Preload vectorizer to avoid fitting it multiple times
vectorizer = TfidfVectorizer()

# Warm-up NLTK tokenizer
word_tokenize("test query")

def stem_query(query):
    tokens = word_tokenize(query.lower())
    return [stemmer.stem(token) for token in tokens]

def rank_documents(query_tokens, inverted_index):
    pass

def process_query(query, inverted_index):
    start_time = time.time()
    
    query_tokens = stem_query(query)
    ranked_docs = rank_documents(query_tokens, inverted_index)
    
    end_time = time.time()
    response_time = end_time - start_time

    results = []
    for score, doc_id in ranked_docs:
        for token in query_tokens:
            if doc_id in inverted_index.get(token, {}):
                url = inverted_index[token][doc_id][0]
                results.append((url, score))
                break

    # sort results by score in descending order and limit to top 20
    results = sorted(results, key=lambda x: x[1], reverse=True)[:20]

    return results, response_time

def main():
    pass

if __name__ == "__main__":
    main()