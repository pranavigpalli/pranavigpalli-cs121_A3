from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from query_retreival import main_three

searching = Blueprint(__name__, "searching")

@searching.route("/")
def home():
    return render_template("homepage.html") 

@searching.route('/search-results')
def my_form_post():
    query = request.args.get('query') # Getting the query as a string
    print(f'the search query typed is: {query}')
    results, time = main_three(query)
    return render_template("search.html", query=query, time=time, results=results)

