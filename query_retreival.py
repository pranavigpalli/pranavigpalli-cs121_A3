import json
import time
import heapq
from math import log10
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download NLTK data
nltk.download('punkt')

# Load the token locations from the JSON file
with open('token_locations_in_index.json', 'r', encoding='utf-8') as file:
    token_locations_in_index = json.load(file)

# Load the doc_id to URL mapping
doc_id_url = {}
with open('doc_id_url.txt', 'r', encoding='utf-8') as file:
    for line in file:
        doc_id, url = line.strip().split(', ')
        doc_id_url[doc_id] = url

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

def get_postings(token):
    first_letter = token[0]
    file_path = f'index/{first_letter.upper()}.txt'
    if token in token_locations_in_index:
        with open(file_path, 'r', encoding='utf-8') as file:
            file.seek(token_locations_in_index[token])
            line = file.readline().strip()
            if ': ' in line:
                try:
                    return json.loads(line.split(': ', 1)[1])
                except json.JSONDecodeError:
                    print(f"Error decoding JSON for token: {token}")
                    return {}
    return {}

def rank_documents(query_terms):
    top_k = 10
    doc_scores = {}
    total_docs = len(doc_id_url)
    initial_idf = None

    for term in query_terms:
        postings = get_postings(term)
        num_docs_with_term = len(postings)
        if num_docs_with_term == 0:
            continue

        idf_value = log10(total_docs / num_docs_with_term)
        if not initial_idf:
            initial_idf = idf_value
        elif idf_value < initial_idf / 2:
            continue

        for doc_id, (freq, importance) in postings.items():
            log_tf = log10(freq + 1)
            score = log_tf * idf_value
            if importance == 1:
                score *= 2  # Boost score for important text
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score

    priority_queue = []
    for doc_id, score in doc_scores.items():
        heapq.heappush(priority_queue, (score, doc_id))
        if len(priority_queue) > top_k:
            heapq.heappop(priority_queue)

    ranked_documents = sorted(priority_queue, reverse=True)
    return ranked_documents

def get_closest_match(query_word):
    if query_word in token_locations_in_index:
        return query_word
    else:
        corrected_word = Word(query_word).correct()
        if corrected_word in token_locations_in_index:
            return corrected_word
    return None

def process_query(query):
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
        corrected_token = get_closest_match(token)
        corrected_tokens.append(corrected_token)

    if not corrected_tokens:
        print("No valid terms found in query.")
        return [], 0

    # print(f"Processed Query Terms: {corrected_tokens}")  # debugging statement

    ranked_docs = rank_documents(corrected_tokens)

    end_time = time.time()
    response_time = end_time - start_time

    results = []
    for score, doc_id in ranked_docs:
        url = doc_id_url[doc_id]
        results.append((url, score))

    return results, response_time

def warm_up():
    """Run a dummy query to initialize everything."""
    process_query("warm up query")

def main():
    # Warm up before taking real queries
    warm_up()

    print("\nWelcome to the search engine!\n")

    while True:
        query = input("Enter your query (type 'exit' to quit the program): ")
        if query == 'exit':
            break
        results, response_time = process_query(query)
        for url, score in results:
            print(f"\t{url}")
        print(f"Response time: {response_time:.5f} seconds\n")

if __name__ == "__main__":
    main()