## FOR M3, OPTIMIZED VERSION OF INVERTED_INDEX
import re      # regex, for extracting tokens
import nltk    # natural language toolkit, for stemming and tokenization
import json    # json, for data storage
import os      # os, for file operations
from nltk.stem import PorterStemmer  # PorterStemmer, for stemming words
from pathlib import Path  # Path, for handling file system paths
from collections import defaultdict  # defaultdict, for creating the inverted index
from bs4 import BeautifulSoup  # BeautifulSoup, for parsing HTML content
from bs4 import XMLParsedAsHTMLWarning  # Warning filter for parsing XML
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

nltk.download('punkt') # Download NLTK tokenizer data

# tokenize() extracts all word tokens from a given text using regex.
# It converts all tokens to lowercase to maintain consistency.
# Time Complexity: O(n), where n is the number of characters in the text.
def tokenize(text): 
    return re.findall(r'\b\w+\b', text.lower())

# stem_tokens() applies Porter stemming to a list of tokens.
# This reduces words to their root form to improve search consistency.
# Time Complexity: O(n * k), where:
#   n = number of tokens
#   k = average length of a token (since stemming processes each character)
def stem_tokens(tokens, stemmer):
    return [stemmer.stem(token) for token in tokens]

# process_file() reads a JSON file containing a webpage, extracts its text content,
# tokenizes and stems it, and identifies important terms from headers and bold/strong tags.
# Returns:
#   - stemmed_tokens: List of processed tokens from the page content.
#   - important_tokens: Set of tokens considered important due to formatting (headers, bold, etc.).
#   - url: URL of the page (from the JSON file metadata).
# Time Complexity: O(f + h + r + n * k), where:
#   f = file size (reading JSON)
#   h = HTML content length (parsing with BeautifulSoup)
#   r = extracted raw text length
#   n = number of tokens processed
def process_file(file_path, stemmer):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        content = data['content']
        url = data['url']  # Extract the URL from the JSON file
        soup = BeautifulSoup(content, 'lxml') # Parse HTML
        text = soup.get_text() # Extract text content
        tokens = tokenize(text) # Tokenize text
        stemmed_tokens = stem_tokens(tokens, stemmer) # Apply stemming

        # Identify important tokens
        important_tokens = set()
        for tag in soup.find_all(['b', 'strong', 'h1', 'h2', 'h3', 'title']):
            important_tokens.update(stem_tokens(tokenize(tag.get_text()), stemmer))

        return stemmed_tokens, important_tokens, url  # Return tokens, important tokens, and URL

# is_important() checks if an HTML tag is considered important for weighting in ranking.
# It helps prioritize headers and bold text for search relevance.
# Time Complexity: O(1) (Checking a fixed set)
def is_important(tag):
    return tag.name in ['b', 'strong', 'h1', 'h2', 'h3', 'title']

# write_index_to_files() saves the inverted index to separate files by first letter.
# If a file exists, it merges new postings with the existing data.
# Time Complexity:
#   - O(m) for reading and parsing existing data (where m = number of tokens in the file)
#   - O(n) for writing back the updated index (where n = number of new tokens)
def write_index_to_files(index, index_dir):
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)

    for letter in 'abcdefghijklmnopqrstuvwxyz':
        file_path = f'{index_dir}/{letter.upper()}.txt'
        existing_data = {}

        # Read existing data from the file
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    token, postings = line.strip().split(': ', 1)
                    existing_data[token] = json.loads(postings)

        # Merge new data with existing data
        for token, postings in index[letter].items():
            if token in existing_data:
                existing_data[token].update(postings)
            else:
                existing_data[token] = postings

        # Write the combined data back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            for token, postings in existing_data.items():
                file.write(f'{token}: {json.dumps(postings)}\n')

# create_report() generates a summary of the indexed documents.
# Time Complexity: O(1) (writing a small fixed number of lines)
def create_report(num_docs, unique_words, index_size, report_path):
    with open(report_path, 'w', encoding='utf-8') as file:
        file.write("Number of indexed documents: {}\n".format(num_docs))
        file.write("Number of unique words: {}\n".format(unique_words))
        file.write("Total size of the index on disk (KB): {:.2f}\n".format(index_size / 1024))

# build_inverted_index() processes all JSON files in the given input directory,
# tokenizing, stemming, and building an inverted index.
# Tokens are grouped by their first letter for efficient storage.
# Time Complexity: O(n * m * T), where:
#   - n = number of subdirectories
#   - m = number of files per subdirectory
#   - T = number of tokens in each file
def build_inverted_index(input_dir):
    stemmer = PorterStemmer()
    inverted_index = defaultdict(lambda: defaultdict(dict))
    token_locations_in_index = {}
    unique_words = set()
    doc_count = 0

    input_path = Path(input_dir)
    for folder in input_path.iterdir():
        if folder.is_dir():
            for file_path in folder.iterdir():
                print(f'{file_path}')
                if file_path.suffix == '.json':
                    doc_count += 1
                    tokens, important_tokens, url = process_file(file_path, stemmer)
                    token_freq = defaultdict(int)

                    # Count term frequency in the document
                    for token in tokens:
                        token_freq[token] += 1
                        unique_words.add(token)
                    
                    # Store tokens in the index, grouped by first letter
                    for token, freq in token_freq.items():
                        first_letter = token[0]
                        importance = 1 if token in important_tokens else 0
                        inverted_index[first_letter][token][doc_count] = [freq, importance]

                    # Store document ID to URL mapping
                    with open('doc_id_url.txt', 'a', encoding='utf-8') as doc_id_url_file:
                        doc_id_url_file.write(f'{doc_count}, {url}\n')

                    # Write index to files after every 5000 documents
                    if doc_count % 5000 == 0:
                        write_index_to_files(inverted_index, 'index')
                        inverted_index = defaultdict(lambda: defaultdict(dict))

    # Write the remaining index to files
    write_index_to_files(inverted_index, 'index')

    # Create token_locations_in_index
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        with open(f'index/{letter}.txt', 'r', encoding='utf-8') as file:
            while True:
                position = file.tell()  # Get current file position
                line = file.readline()
                if not line:
                    break
                token = line.split(':')[0]
                token_locations_in_index[token] = position

    with open('token_locations_in_index.json', 'w', encoding='utf-8') as file:
        json.dump(token_locations_in_index, file)

    create_report(doc_count, len(unique_words), sum(os.path.getsize(f'index/{letter.upper()}.txt') for letter in 'abcdefghijklmnopqrstuvwxyz'), 'report.txt')

# clear_files() resets the indexing system by truncating or removing stored index files.
# Time Complexity: O(n), where n is the number of files being cleared.
def clear_files():
    # Clear index directory
    index_dir = Path('index')
    for file in index_dir.iterdir():
        if file.is_file():
            with open(file, 'w', encoding='utf-8') as f:
                f.truncate(0)

    # Clear doc_id_url.txt
    with open('doc_id_url.txt', 'w', encoding='utf-8') as file:
        file.truncate(0)

    # Clear token_locations_in_index.json
    with open('token_locations_in_index.json', 'w', encoding='utf-8') as file:
        file.truncate(0)

if __name__ == "__main__":
    clear_files()
    build_inverted_index('DEV')