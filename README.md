# JPAGE Search Engine!
# Assignment 3 for Professor Ahmed's CS121/INF141 Winter 2025

Pranavi Gollanapalli, Gauresh Gururaj, Audrey Lu, Emily Hames, Jasmyn Villanueva

## Background

This is our search engine for CS 121 / INF 141 in Winter 2025 with Professor Iftekhar Ahmed.
JPAGE Search Engine searches through 55,000+ HTML documents and has a query response time under 100ms.

## Dependencies

Please download collections, regex, nltk, BeautifulSoup, textblob, scikit-learn, and flask using pip install.

## Launch the Program

### M1: Build Index
Download the [DEV folder](https://www.ics.uci.edu/~algol/teaching/informatics141cs121w2022/a3files/developer.zip) into your current directory. Ensure you have a folder named "index" in your current directory that is either empty or contains a previously created inverted index created by inverted_index_m3.py. Run inverted_index_m3.py. This will read through the files in the DEV folder, which contain JSON files with data about UCI ICS, Statistics, and Informatics websites, and then parse that data to create an inverted index. The inverted index is inside the "index" folder with .txt files labeled A.txt through Z.txt.

### M2: Search and Retrieval
Ensure the steps for M1 are already completed. Run query_retreival.py. Enter search queries or type "exit" to end the program. This program uses the inverted index created in M1 to return the top 10 URLs for the inputted query.

### M3: Completed Search Engine
Ensure the steps for M1 and M2 are already completed. Run app.py. After the app has started, go to http://127.0.0.1:8000/searching. Here, you can enter queries into the search engine and view the top 10 results from your search. 

Features in the search engine that optimize the results include:
- Closest term matching for mispelled words or rare terms using TextBlob
- Increasing the weight of words in page titles
- Using stop word filtering for long queries
- Using precomputed data for the most common queries