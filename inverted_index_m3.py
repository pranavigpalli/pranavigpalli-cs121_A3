## FOR M3, OPTIMIZED VERSION OF INVERTED_INDEX
import re
import nltk
import json
import os
from nltk.stem import PorterStemmer
from pathlib import Path
from collections import defaultdict
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

nltk.download('punkt')

def tokenize(text): 
    return re.findall(r'\b\w+\b', text.lower())

def stem_tokens(tokens, stemmer):
    return [stemmer.stem(token) for token in tokens]

def process_file(file_path, stemmer):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        content = data['content']
        url = data['url']  # Extract the URL from the JSON file
        soup = BeautifulSoup(content, 'lxml')
        text = soup.get_text()
        tokens = tokenize(text)
        stemmed_tokens = stem_tokens(tokens, stemmer)

        # Identify important tokens
        important_tokens = set()
        for tag in soup.find_all(['b', 'strong', 'h1', 'h2', 'h3', 'title']):
            important_tokens.update(stem_tokens(tokenize(tag.get_text()), stemmer))

        return stemmed_tokens, important_tokens, url  # Return tokens, important tokens, and URL

def is_important(tag):
    return tag.name in ['b', 'strong', 'h1', 'h2', 'h3', 'title']

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

def create_report(num_docs, unique_words, index_size, report_path):
    with open(report_path, 'w', encoding='utf-8') as file:
        file.write("Number of indexed documents: {}\n".format(num_docs))
        file.write("Number of unique words: {}\n".format(unique_words))
        file.write("Total size of the index on disk (KB): {:.2f}\n".format(index_size / 1024))

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

                    for token in tokens:
                        token_freq[token] += 1
                        unique_words.add(token)

                    for token, freq in token_freq.items():
                        first_letter = token[0]
                        importance = 1 if token in important_tokens else 0
                        inverted_index[first_letter][token][doc_count] = [freq, importance]

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