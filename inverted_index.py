import re		# regex, for filtering out tokens
import nltk	    # natural language toolkit, for nltk.stem and nltk.download(‘punkt’)
import json	    # json, for data storage
import os		# os, for checking if a file exists
from nltk.stem import PorterStemmer		# PorterStemmer, for Porter stemming
from pathlib import Path		        # Path, for processing files and directories
from collections import defaultdict		# defaultdict, for our inverted index
from bs4 import BeautifulSoup		    # BeautifulSoup, to turn text into a soup object

nltk.download('punkt')      # to increase speed of creating the index

#	tokenize() takes in a string/text and uses regex to filter out all 
#   possible tokens from that text, in lowercase form. returns all tokens found.
#	Time Complexity: O(n)
def tokenize(text): 
    return re.findall(r'\b\w+\b', text.lower())

#	stem_tokens() takes the tokens and a PorterStemmer, and returns 
#   a list of the stem words from every token.  
#	Time Complexity: O(n * k), where k is the average token length, 
#   and n is the amount of tokens 
def stem_tokens(tokens, stemmer):
    return [stemmer.stem(token) for token in tokens]

#	process_file() takes a file path and the PorterStemmer, and opens the file, 
#	turn its content into a json, turns that content into a soup object, retrieves its text, 
#	tokenizes it, and then turns the tokens into stemmed tokens and returns that.
#	Time Complexity: O(f + 2h + r + n * k), where:
#   f is file size, 
#   n is number of tokens
#   h is the. length of HTML content
#   r is the length of the text
def process_file(file_path, stemmer):
    with open(file_path, 'r', encoding='utf-8') as file:    #	O(f)
        data = json.load(file)  #	O(f)
        content = data['content']
        soup = BeautifulSoup(content, 'lxml')   #	O(h)
        text = soup.get_text()      #	O(h)
        tokens = tokenize(text)     #	O(r)
        stemmed_tokens = stem_tokens(tokens, stemmer)   #	O(n * k)
        return stemmed_tokens


# updates an inverted index stored in a JSON file.
# Time complexity is worst case O(ND) given:
# D be the number of documents per token
# N be the number of tokens already in existing_index
# This only occurs if every token is unique, so the real
# time complexity will be lower
def write_index_to_file(index, file_path):
    # checks if file exists
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            existing_index = json.load(file)
    else:
        existing_index = {}

    # merge new index with existing one
    for token, postings in index.items():
        if token in existing_index:
            for doc_id, freq in postings.items():
                if doc_id in existing_index[token]:
                    # if the document exists, add the frequency
                    existing_index[token][doc_id] += freq
                else:
                    existing_index[token][doc_id] = freq
        else:
            # if a token is new, add it to existing_index
            existing_index[token] = postings

    # write the updated index back to the file.
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_index, file)

# generates a simple text report about the indexing process.
# O(1) time complexity as it simply writes the already
# processed variables into the report document
def create_report(num_docs, unique_words, index_size, report_path):
    with open(report_path, 'w', encoding='utf-8') as file:
        file.write("Number of indexed documents: {}\n".format(num_docs))
        file.write("Number of unique words: {}\n".format(unique_words))
        file.write("Total size of the index on disk (KB): {:.2f}\n".format(index_size / 1024))

def build_inverted_index(input_dir):
	# This function takes in an input directory containing the
	# documents we need to process.
    # Overall Time Complexity: O(n*m*T), where n = num subdirectories, m = num files/subdirectory, T = num tokens in file

    stemmer = PorterStemmer()
	# Creates a PorterStemmer instance to stem words.

    inverted_index = defaultdict(dict)
	# This initializes an inverted index as a defaultdict
	# with a default value of an empty list. 
	# The keys are expected to be stemmed words, and the values
	# are lists of tuples with filename and frequency for word
	# frequencies. 

    doc_count = 0
	# doc_count is initialized as 0 and this variable tracks
	# how many documents are being processed.

    unique_words = set()
	# unique_words is a set storing unique words found in all the documents. 

    index_file_path = 'inverted_index.json'
	# This is the path storing our inverted index

    report_file_path = 'report.txt'
	# This is the path storing our summary report to write

    if os.path.exists(index_file_path):
        os.remove(index_file_path)
	# This if statement checks if our inverted index file exists already, and if it does, it deletes the
	# existing file to start over. 

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
	# This checks if there is remaining data in inverted_index
	# If there is data, it writes everything into index_file_path
	# by calling write_index_to_file

    index_size = os.path.getsize(index_file_path)
	# This retrieves the size in bytes of inverted_index.json after writing into it

    create_report(doc_count, len(unique_words), index_size, report_file_path)
	# This calls create_report, which generates our final report

if __name__ == "__main__":
    build_inverted_index('DEV')