from flask import Flask
from searching import searching
app = Flask(__name__)
app.register_blueprint(searching, url_prefix="/searching")

if __name__ == '__main__':
    print('restarting flask app')
    app.run(debug=True, port=8000)
