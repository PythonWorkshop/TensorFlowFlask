from flask import Flask
from flask import request, Response
from flask import abort

import json
import os

app = Flask(__name__)

@app.errorhandler(401)
def custom_401(error):
    return Response('Access Unauthorized', 401, {'WWWAuthenticate':'Basic realm="Login Required"'})

@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.config['DEBUG'] = True
    app.run()

