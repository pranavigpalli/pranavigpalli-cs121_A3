from flask import Blueprint, render_template, request
from query_retreival import main_three

searching = Blueprint(__name__, "searching")

'''Function for rendering the start/homepage'''
@searching.route("/")
def home():
    return render_template("homepage.html") 

'''Function for rendering the result page'''
@searching.route('/search-results')
def my_form_post():
    query = request.args.get('query') # Getting the query as a string
    # print(f'the search query typed is: {query}')
    results, time = main_three(query)
    return render_template("search.html", query=query, time=time, results=results)

