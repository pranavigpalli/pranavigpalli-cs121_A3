import json
import time
import heapq
import math
from math import log10
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# threshold for close matches
THRESHOLD = 60

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

with open("stop_words.txt") as f:
    stop_words = set(f.read().split())

def stem_query(query):
    tokens = word_tokenize(query.lower())
    return [stemmer.stem(token) for token in tokens]

def get_tf_idf_vectors(query_tokens, inverted_index):
    doc_vectors = defaultdict(lambda: [0] * len(query_tokens))
    query_vector = [0] * len(query_tokens)
    N = len(inverted_index)

    for i, token in enumerate(query_tokens):
        if token in inverted_index:
            df = len(inverted_index[token])
            idf = math.log(N / (df + 1))
            query_vector[i] = idf

            for doc_id, (url, tf) in inverted_index[token].items():
                doc_vectors[doc_id][i] = tf * idf

    return query_vector, doc_vectors

def rank_documents(query_terms, inverted_idx):
    top_k = 10
    
    term_postings_list = []
    for term in query_terms:
        if term in inverted_idx:
            term_postings_list.append(inverted_idx[term])

    doc_scores = {}
    total_docs = len(inverted_idx)
    initial_idf = None
    for i in range(len(term_postings_list)):
        postings = term_postings_list[i]
        num_docs_with_term = len(postings)
        if num_docs_with_term > 0:
            idf_value = log10(total_docs / num_docs_with_term)
            if initial_idf is None:
                initial_idf = idf_value
            # Exclude terms that are too common, prioritize terms that are more unique
            if idf_value >= initial_idf / 3:
                for doc_id, (url, term_freq) in postings.items():
                    log_tf = log10(1 + term_freq)
                    doc_scores[doc_id] = log_tf * idf_value + doc_scores.get(doc_id, 0)

    min_heap_scores = []  # min-heap to keep track of top k scores
    for doc_id in doc_scores:
        score = doc_scores[doc_id]
        heapq.heappush(min_heap_scores, (score, doc_id))
        if len(min_heap_scores) > top_k:
            heapq.heappop(min_heap_scores)

    ranked_documents = sorted(min_heap_scores, reverse=True)
    return ranked_documents

def get_closest_match(query_word, index):
    if query_word in index:
        return query_word
    else:
        corrected_word = Word(query_word).correct()
        if corrected_word in index:
            return corrected_word
    return None

def process_query(query, inverted_index):
    start_time = time.time()
    
    query_tokens = word_tokenize(query.lower())
    
    # Remove stop words if query is long
    if len(query_tokens) >= 5:
        query_tokens = [token for token in query_tokens if token not in stop_words]

    # Stem all tokens
    stemmed_tokens = [stemmer.stem(token) for token in query_tokens]

    # Handle misspellings
    corrected_tokens = []
    for token in stemmed_tokens:
        corrected_token = get_closest_match(token, inverted_index)
        corrected_tokens.append(corrected_token)

    if not corrected_tokens:
        print("No valid terms found in query.")
        return [], 0

    print(f"Processed Query Terms: {corrected_tokens}")

    ranked_docs = rank_documents(corrected_tokens, inverted_index)

    end_time = time.time()
    response_time = end_time - start_time

    results = []
    for score, doc_id in ranked_docs:
        for token in corrected_tokens:
            if doc_id in inverted_index.get(token, {}):
                url = inverted_index[token][doc_id][0]
                results.append((url, score))
                break

    return results, response_time



def warm_up():
    """Run a dummy query to initialize everything."""
    process_query("warm up query", inverted_index)

def main():
    # Warm up before taking real queries
    warm_up()

    print("\nWelcome to the search engine!\n")

    while True:
        query = input("Enter your query (type 'exit' to quit the program): ")
        if query == 'exit':
            break
        results, response_time = process_query(query, inverted_index)
        for url, score in results:
            print(f"\t{url}")
        print(f"Response time: {response_time:.5f} seconds\n")

if __name__ == "__main__":
    main()
