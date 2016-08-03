from flask import Flask
from flask import request, Response, render_template, url_for, redirect
from flask import abort
from flask_wtf.csrf import CsrfProtect
# restore trained data
import tensorflow as tf

import sys

sys.path.append('wine_quality')
import wine_quality.model as model
from wine_quality.training import train_model
import json
import os
from form import TestParameterForm, TrainingDataForm
from werkzeug.utils import secure_filename
import pandas as pd


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
        # simple([[0.7, 0, 1.9, 0.076, 11, 34, 0.99780, 3.51, 0.56, 9.4]])
        results = simple([[0.7, 0, 1.9, 0.076, 11, 34, 0.99780, 3.51, 0.56, 9.4]])
        return render_template('test_parameters.html', form=form, result=results[0])
    return render_template('test_parameters.html', form=form)


@app.route('/train/', methods=('GET', 'POST'))
def upload():
    form = TrainingDataForm()
    if form.validate_on_submit():
        model_name = form.model_name.data
        learning_rate = float(form.learning_rate.data)
        batch_size = int(form.batch_size.data)
        filename = secure_filename(form.training_data.data.filename)
        print(form.__dict__)
        # Save to Redis here
        form.training_data.data.save('wine_quality/data/' + filename)
        dataframe = pd.read_csv('wine_quality/data/' + filename, sep=';')
        train_model(dataframe, learning_rate, batch_size)
    else:
        filename = None
    return render_template('test_data_upload.html', form=form, filename=filename)


if __name__ == '__main__':
    app.config['DEBUG'] = True
    app.config['SECRET_KEY'] = "SOME SECRET KEY HERE"
    app.config['WTF_CSRF_ENABLED'] = True
    app.run()


