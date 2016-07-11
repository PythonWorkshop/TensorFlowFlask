from flask import Flask
from flask import request, Response, render_template
from flask import abort
from flask_wtf.csrf import CsrfProtect

csrf = CsrfProtect()

import json
import os
from form import TestParameterForm


app = Flask(__name__)
csrf.init_app(app)



@app.errorhandler(401)
def custom_401(error):
    return Response('Access Unauthorized', 401, {'WWWAuthenticate':'Basic realm="Login Required"'})


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/test/', methods=['GET', 'POST'])
def test_parameters():
    form = TestParameterForm(request.form)
    if request.method == 'POST' and form.validate():
        print("Alcohol Percentage: {}".format(form.alcohol_percentage))
        print("Fixed Acidity: {}".format(form.fixed_acidity))
        return redirect(url_for('hello_world'))
    return render_template('test_parameters.html', form=form)


if __name__ == '__main__':
    app.config['DEBUG'] = True
    app.config['SECRET_KEY'] = "SOME SECRET KEY HERE"
    app.config['WTF_CSRF_ENABLED'] = True
    app.run()

