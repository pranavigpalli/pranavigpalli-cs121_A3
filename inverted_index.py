import re
import nltk
import json
from nltk.stem import PorterStemmer
from pathlib import Path
from collections import defaultdict
import os
from bs4 import BeautifulSoup

nltk.download('punkt')

#  tokenize() takes in a string/text and uses regex to filter out all possible tokens from that text, in lower case form.
#  Returns all tokens found.
#  Time Complexity: 
def tokenize(text): 
    return re.findall(r'\b\w+\b', text.lower())


def stem_tokens(tokens, stemmer):
    return [stemmer.stem(token) for token in tokens]

def process_file(file_path, stemmer):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        content = data['content']
        soup = BeautifulSoup(content, 'lxml')
        text = soup.get_text()
        tokens = tokenize(text)
        stemmed_tokens = stem_tokens(tokens, stemmer)
        return stemmed_tokens

def write_index_to_file(index, file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            existing_index = json.load(file)
    else:
        existing_index = {}

    for token, postings in index.items():
        if token in existing_index:
            for doc_id, freq in postings.items():
                if doc_id in existing_index[token]:
                    existing_index[token][doc_id] += freq
                else:
                    existing_index[token][doc_id] = freq
        else:
            existing_index[token] = postings

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_index, file)

def create_report(num_docs, unique_words, index_size, report_path):
    with open(report_path, 'w', encoding='utf-8') as file:
        file.write("Number of indexed documents: {}\n".format(num_docs))
        file.write("Number of unique words: {}\n".format(unique_words))
        file.write("Total size of the index on disk (KB): {:.2f}\n".format(index_size / 1024))

def build_inverted_index(input_dir):
    stemmer = PorterStemmer()
    inverted_index = defaultdict(dict)
    doc_count = 0
    unique_words = set()
    index_file_path = 'inverted_index.json'
    report_file_path = 'report.txt'

    if os.path.exists(index_file_path):
        os.remove(index_file_path)

    #Time complexity for overall nested for loop O(n*m*T), where n = num subdirectories, m = num files/subdirectory, T = num tokens in file
    #Lists all the files and directories in input_dir
    # O(n), where n = number of subdirectories in input_dir
    for folder in Path(input_dir).iterdir():
        
	    #Only allows the directories to pass through
	    # O(m), where m = number of files per subdirectory
        if folder.is_dir():
            folder_name = folder.name
            #Lists all files in the directory
            for file in folder.iterdir():
                print(f"{folder_name}/{file.name}")
                #Only allows the .json files to pass through
                if file.is_file() and file.suffix == '.json':

                    #doc_count is incremented for each file processed
                    doc_count += 1

                    #Tokenize the text in the file
                    tokens = process_file(file, stemmer)
                    #Create a tracker for each token’s frequency
                    token_freq = defaultdict(int)

                    #Count token frequency 
                    # O(T), where T is the number of tokens in the file
                    for token in tokens:
                        token_freq[token] += 1
                        unique_words.add(token)

                    #For each token we add an entry to the inverted_index which tracks info about the tokens
                    # O(U), where U = number of unique tokens in the file
                    for token, freq in token_freq.items():
                        inverted_index[token][file.name] = freq
                    # Every 5000 tokens we write the current inverted_index to a file and clear the memory
                    if doc_count % 5000 == 0:
                        write_index_to_file(inverted_index, index_file_path)
                        inverted_index.clear()

    if inverted_index:
        write_index_to_file(inverted_index, index_file_path)

    index_size = os.path.getsize(index_file_path)
    create_report(doc_count, len(unique_words), index_size, report_file_path)

if __name__ == "__main__":
    build_inverted_index('DEV')