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

# Load precomputed results
try:
    with open('precomputed_results.json', 'r', encoding='utf-8') as file:
        precomputed_results = json.load(file)
except FileNotFoundError:
    precomputed_results = {}

# Initialize the stemmer
stemmer = PorterStemmer()

# Preload vectorizer to avoid fitting it multiple times
vectorizer = TfidfVectorizer()

# Warm-up NLTK tokenizer
word_tokenize("test query")

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
            idf = math.log(N / (df + 1))  # Avoid division by zero
            query_vector[i] = idf

            for doc_id, (url, tf) in inverted_index[token].items():
                doc_vectors[doc_id][i] = tf * idf

    return query_vector, doc_vectors

def rank_documents(query_tokens, inverted_index):
    k = 20
    accumulator = {}  # Map: doc_id ==> tf.idf
    term_collections = []
    pq = []
    c_length = len(inverted_index)

    for term in query_tokens:
        if term in inverted_index:
            term_collections.append(inverted_index[term])
        else:
            print(f'term "{term}" not found')

    first_idf = None
    for i, term_postings in enumerate(term_collections):
        num_docs = len(term_postings)
        if num_docs == 0:
            continue

        idf = log10(c_length / num_docs)
        if not first_idf:
            first_idf = idf
        elif idf < first_idf / 2:
            continue

        for doc_id, (url, tf) in term_postings.items():
            log_tdf = log10(tf + 1)
            accumulator[doc_id] = accumulator.get(doc_id, 0) + log_tdf * idf

    for doc_id, score in accumulator.items():
        heapq.heappush(pq, (score, doc_id))
        if len(pq) > k:
            heapq.heappop(pq)

    ranked_docs = sorted(pq, reverse=True)
    return ranked_docs

def process_query(query, inverted_index):
    start_time = time.time()

    if query in precomputed_results:
        print("Fetching query results from cache...")
        return precomputed_results[query], 0
    
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

    # Sort results by score in descending order and limit to top 20
    results = sorted(results, key=lambda x: x[1], reverse=True)[:20]

    # Storing results for later searches
    precomputed_results[query] = results
    with open('precomputed_results.json', 'w', encoding='utf-8') as file:
        json.dump(precomputed_results, file, indent=4)

    return results, response_time

def check_index_update():
    """Checks if the index has changed and clears cache if necessary."""

    print("Checking if index changed...")
    
    with open("inverted_index.json", "r") as f:
        current_index = f.read()

    try:
        with open("last_index_snapshot.json", "r") as f:
            previous_index = f.read()
    except FileNotFoundError:
        previous_index = ""

    if current_index != previous_index:
        print("Index has changed. Clearing cached results...")
        global precomputed_results
        precomputed_results = {}  # Clear cache
        with open('precomputed_results.json', 'w', encoding='utf-8') as file:
            json.dump({}, file)
        with open("last_index_snapshot.json", "w") as f:
            f.write(current_index)

def warm_up():
    """Run a dummy query to initialize everything."""
    process_query("warm up query", inverted_index)

def main():
    # Checking if the index has updated
    check_index_update()

    # Warm up before taking real queries
    warm_up()

    while True:
        query = input("Enter your query: ")
        if query == 'exit':
            break
        results, response_time = process_query(query, inverted_index)
        for url, score in results:
            print(f"\t{url}")
        print(f"Response time: {response_time:.5f} seconds")

if __name__ == "__main__":
    main()