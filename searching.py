from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from query_retreival import main_three

searching = Blueprint(__name__, "searching")

@searching.route("/")
def home():
    return render_template("homepage.html") 

def spector(str):
    print(str)

@searching.route('/search-results')
def my_form_post():
    query = request.args.get('query') # Getting the query as a string
    spector(f'the search query typed is: {query}') # spector() can be replaced with any other function out there
    results, time = main_three(query)
    return render_template("search.html", query=query, time=time, results=results)

"""
TO-DO:
- integrate m2 (& turn m2 into something workable for m3)
- show results in search.html (from python list...?)
- show the past query as editable & not as a preview
- get a domain
- fix the logo in the results lol
"""
