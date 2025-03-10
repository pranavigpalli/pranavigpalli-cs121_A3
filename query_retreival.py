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

# Load the token locations from the JSON file maps each token to its position in the index
with open('token_locations_in_index.json', 'r', encoding='utf-8') as file:
    token_locations_in_index = json.load(file)

# Load the doc_id to URL mapping associates document IDs with URLs
doc_id_url = {}
with open('doc_id_url.txt', 'r', encoding='utf-8') as file:
    for line in file:
        doc_id, url = line.strip().split(', ')
        doc_id_url[doc_id] = url

# Load the inverted index from the JSON file
# with open('inverted_index.json', 'r', encoding='utf-8') as file:
#     inverted_index = json.load(file)

# Load precomputed results
try:
    with open('precomputed_results.json', 'r', encoding='utf-8') as file:
        precomputed_results = json.load(file)
except FileNotFoundError:
    precomputed_results = {}

# Initialize the stemmer for reducing words to their root forms 
stemmer = PorterStemmer()

# Preload vectorizer to avoid fitting it multiple times used for TF-IDF calculations
vectorizer = TfidfVectorizer()

# Warm-up NLTK tokenizer ensures it's ready for use without initial delay
word_tokenize("test query")

# Load stop words from a file to remove common words from queries
with open("stop_words.txt") as f:
    stop_words = set(f.read().split())

def stem_query(query):
    """Tokenizes and stems a given query to standardize word forms."""
    tokens = word_tokenize(query.lower()) # Convert to lowercase and tokenize
    return [stemmer.stem(token) for token in tokens] # Apply stemming

def get_postings(token):
    """Retrieves the postings list for a given token from the pre-built index file."""
    first_letter = token[0]  # Identify which file the token is stored in
    file_path = f'index/{first_letter.upper()}.txt'
    if token in token_locations_in_index:
        with open(file_path, 'r', encoding='utf-8') as file:
            file.seek(token_locations_in_index[token]) # Jump to the token’s position
            line = file.readline().strip()
            if ': ' in line:
                try:
                    return json.loads(line.split(': ', 1)[1]) # Extract and parse JSON
                except json.JSONDecodeError:
                    print(f"Error decoding JSON for token: {token}")
                    return {} 
    return {} # Return empty if token is not found

def rank_documents(query_terms):
    """Ranks documents based on TF-IDF scores for the given query terms."""
    top_k = 10  # Limit the number of top-ranked documents
    doc_scores = {}  # Store document scores
    total_docs = len(doc_id_url)  # Total number of indexed documents
    initial_idf = None  # Store the IDF of the first term to compare others

    for term in query_terms:
        postings = get_postings(term)  # Retrieve documents containing the term
        num_docs_with_term = len(postings)  # Get the number of documents containing the term
        if num_docs_with_term == 0:
            continue # Skip terms that are not in the index

        idf_value = log10(total_docs / num_docs_with_term) # Compute IDF
        if not initial_idf:
            initial_idf = idf_value
        elif idf_value < initial_idf / 2:
            continue # Skip terms with very low IDF meaning too common

        for doc_id, (freq, importance) in postings.items():
            log_tf = log10(freq + 1) # Compute log(TF)
            score = log_tf * idf_value  # Compute TF-IDF score
            if importance == 1:
                score *= 2  # Boost score for important text
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score # Accumulate scores

    # Use a min-heap to efficiently track top-ranked documents
    priority_queue = []
    for doc_id, score in doc_scores.items():
        heapq.heappush(priority_queue, (score, doc_id))
        if len(priority_queue) > top_k:
            heapq.heappop(priority_queue) # Remove the lowest-ranked document

    # Sort documents in descending order of score
    ranked_documents = sorted(priority_queue, reverse=True)
    return ranked_documents

def get_closest_match(query_word):
    """Returns the closest matching word from the index if the input word is misspelled."""
    if query_word in token_locations_in_index:
        return query_word # Return original word if found
    else:
        corrected_word = Word(query_word).correct() # Attempt to correct spelling
        if corrected_word in token_locations_in_index: 
            return corrected_word # Return corrected word if found
    return None # Return None if no match is found

def process_query(query):
    """Processes the user's query by tokenizing, stemming, handling misspellings, and ranking results."""
    start_time = time.time() # Start timing query processing

    # If the search query is precomputed, it's fetched from cache
    if query in precomputed_results:
        print("Fetching query results from cache...")
        return precomputed_results[query], 0
    
    query_tokens = word_tokenize(query.lower()) # Tokenize and lowercase query
    
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
        return [], 0 # Return empty results if no valid terms are found

    # print(f"Processed Query Terms: {corrected_tokens}")  # debugging statement

    ranked_docs = rank_documents(corrected_tokens)

    end_time = time.time()  # Stop timing query processing
    response_time = end_time - start_time

    # Format the search results with URLs and scores
    results = []
    for score, doc_id in ranked_docs:
        url = doc_id_url[doc_id]
        results.append((url, score))

    return results, response_time # Return results and query response time

def warm_up():
    """Run a dummy query to initialize everything."""
    process_query("warm up query")

def main():
    """Main function to run the search engine in an interactive mode."""
    # Warm up before taking real queries
    warm_up()

    print("\nWelcome to the search engine!\n")

    while True:
        query = input("Enter your query (type 'exit' to quit the program): ")
        if query == 'exit':
            break
        results, response_time = process_query(query) # Process the user’s search query
        for url, score in results:
            print(f"\t{url}")
        print(f"Response time: {response_time:.5f} seconds\n") # Show query execution time

def main_three(query):
    """Handles query processing for M3's UI by returning search results."""
    warm_up()  # Warm up before taking real queries
    results, response_time = process_query(query)
    return results, response_time

if __name__ == "__main__":
    main()