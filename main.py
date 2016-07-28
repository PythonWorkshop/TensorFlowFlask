from flask import Flask
from flask import request, Response, render_template, url_for, redirect
from flask import abort
from flask_wtf.csrf import CsrfProtect
# restore trained data
import tensorflow as tf

import sys
import wine_quality.model as model
import json
import os
from form import TestParameterForm



x = tf.placeholder("float", [None, 10])
sess = tf.Session()

with tf.variable_scope("softmax_regression"):
    y1, variables = model.softmax_regression(x)
saver = tf.train.Saver(variables)
saver.restore(sess, "wine_quality/data/softmax_regression.ckpt")
def simple(x1):
    return sess.run(y1, feed_dict={x: x1})


csrf = CsrfProtect()
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
        print(form.__dict__)
        print(form.citric_acid.data)
        print(form.citric_acid.data)
        print(form.citric_acid.data)
        print(form.citric_acid.data)
        print(form.citric_acid.data)
        print(form.citric_acid.data)
        # simple([[0.7, 0, 1.9, 0.076, 11, 34, 0.99780, 3.51, 0.56, 9.4]])
        results = simple([[0.7, 0, 1.9, 0.076, 11, 34, 0.99780, 3.51, 0.56, 9.4]])
        return render_template('test_parameters.html', form=form, result=results[0])
    return render_template('test_parameters.html', form=form)


if __name__ == '__main__':
    app.config['DEBUG'] = True
    app.config['SECRET_KEY'] = "SOME SECRET KEY HERE"
    app.config['WTF_CSRF_ENABLED'] = True
    app.run()


