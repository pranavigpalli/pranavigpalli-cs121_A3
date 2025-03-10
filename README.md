# JPAGE Search Engine!
# Assignment 3 for Professor Ahmed's CS121/INF141 Winter 2025

Pranavi Gollanapalli, Gauresh Gururaj, Audrey Lu, Emily Hames, Jasmyn Villanueva

## Background

This is our search engine for CS 121 / INF 141 in Winter 2025 with Professor Iftekhar Ahmed.

## Dependencies

Please download collections, regex, nltk, BeautifulSoup, textblob, scikit-learn, and flask using pip install.

## Launch the Program

### M1: Build Index
First, ensure you have a folder named "index" that is either empty or has previously parsed data. Open the terminal and type "python inverted_index.py". This will open the "DEV" folder, containing JSON files with the HTML data from ICS websites, and then parse that data to create an inverted index. The inverted index is inside the "index" folder with .txt files labeled A through Z.

### M2: Query Retrieval
Ensure the steps for M1 are already completed. Open the terminal and type "python query_retreival.py". Then enter your searches or type exit to end the searching. This code uses the inverted index from M1 to return the top 10 URLs that match the entered query.

### M3: Completed Search Engine
Repeat the steps for M1 using "inverted_index_m3" instead. Complete the steps for M2 as well. Open the terminal and type "python app.py". Then, go to http://127.0.0.1:8000/searching after running the previous command. Here, you can enter your query and view the top 10 results from your search. Some features we have to optimize our search engine include:
- closest term matching for mispelled words or rare terms using TextBlob
- increasing the weight of words in page titles using an updated inverted index
- using stop words filtering for long queries
- using precomputed data for the most common terms