import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from wine_quality.softmax_regression import softmax_regression
from boto.s3.connection import S3Connection


def _remove_outliers(wine_df, threshold, columns):
    """ Removes the outliers from the dataframe based on value greater than threshold
        number of standard deviations """
    for col in columns:
        mask = wine_df[col] > float(threshold) * wine_df[col].std() + wine_df[col].mean()
        wine_df.loc[mask==True, col] = np.nan  # noqa
        mean_property = wine_df.loc[:, col].mean()
        wine_df.loc[mask==True, col] = mean_property  # noqa
    return wine_df


def _dense_to_one_hot(dense_labels, num_classes=2):
    """ Converts dense label data to one-hot encoded vectors """
    num_labels = len(dense_labels)
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + dense_labels] = 1
    return labels_one_hot


def _filter_wine_to_categories(wine_df, categories={'Bad', 'Good'}):
    """ Filters out the rows for which categories are not in category set """
    return wine_df.ix[[item in categories for item in wine_df['category']],:]


def _create_and_separate_bins(wine_df, bins, new_bins, labels, new_labels,
                              categories={'Bad', 'Good'}):
    """ Creates the categories to bin the data into and filters out the rows that
        are not in the categories """
    wine_df['quality_bins'] = pd.cut(wine_df.quality, bins, labels, include_lowest=True)
    wine_df['category'] = pd.cut(wine_df.quality,
                                 new_bins, labels=new_labels, include_lowest=True)
    wine_df = _filter_wine_to_categories(wine_df, categories)

    wine_df['quality_bins'] = pd.cut(wine_df.quality,
                                     bins, labels=['Bad', 'Good'], include_lowest=True)
    return wine_df


def _get_x_one_hot_values(wine_df):
    """ Retrieves the one-hot feature data from the full dataframe """
    X_red_wine = wine_df.iloc[:, 1:-2].get_values()
    return X_red_wine


def _get_y_one_hot_values(wine_df, labels_dict={'Bad': '1', 'Good': '1'}):
    """ Retrieves the one-hot label data from the full dataframe """
    y_red_wine = wine_df[['quality_bins']].get_values()

    y_red_wine_raveled = y_red_wine.ravel()
    for key, value in labels_dict.items():
        y_red_wine_integers = [y.replace(key, value) for y in y_red_wine_raveled]
    y_red_wine_integers = [np.int(y) for y in y_red_wine_integers]

    y_one_hot = _dense_to_one_hot(y_red_wine_integers, num_classes=2)
    return y_one_hot


class tf_model():
    """ This is a class method to enable access to a TensorFlow model. It
        provides helper methods to save and load the core model from local or AWS S3
        storage, and also allows training of the model with input parameters. """

    def __init__(self):
        self.sess = tf.Session()

        with tf.variable_scope("softmax_regression"):
            self.X = tf.placeholder("float", [None, 10])
            self.y1, self.variables = softmax_regression(self.X)
            self.saver = tf.train.Saver(self.variables)

    def predict(self, x1):
        return self.sess.run(self.y1, feed_dict={self.X: x1})

    def save_locally(self, filename='softmax_regression.ckpt'):
        path = self.saver.save(self.sess, '/tmp/wine_quality/' + filename)
        return path

    def load_locally(self, filename='softmax_regression.ckpt'):
        self.saver.restore(self.sess, '/tmp/wine_quality/' + filename)

    def save_to_s3(self, filename, model_name):
        try:
            AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
            AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
            c = S3Connection(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
            b = c.get_bucket('flasktensorflow')  # substitute your bucket name here
            k = b.new_key(model_name)
            f = open(filename, 'rb')
            k.set_contents_from_file(f, encrypt_key=True)
            print("Saving to S3")
        except:
            return False
        return True

    def load_from_s3(self, filename, model_name):
        try:
            AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
            AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
            c = S3Connection(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
            b = c.get_bucket('flasktensorflow')  # substitute your bucket name here
            k = b.Key(b)
            k.key = model_name
            f = open(filename, 'rb')
            k.get(f, encrypt_key=True)
            print("Saving to S3")
        except:
            return False
        return True

    def train(self, training_df, learning_rate=0.001, batch_size=126, model_name="softmax_model", filename="data/softmax_regression.ckpt"):
        column_list = training_df.columns.tolist()
        threshold = 5
        training_df = _remove_outliers(training_df, threshold, column_list[0:-1])

        bins = [3, 5, 6, 8]
        new_bins = [3, 5, 8]
        labels = ['Bad', 'Average', 'Good']
        new_labels = ['Bad', 'Good']
        training_df = _create_and_separate_bins(training_df, bins,
                                                new_bins, labels, new_labels)

        labels_dict = {'Bad': '1', 'Good': '0'}
        y_red_wine = _get_y_one_hot_values(training_df, labels_dict)
        X_red_wine = _get_x_one_hot_values(training_df)

        X_train, X_test, y_train, y_test = train_test_split(X_red_wine, y_one_hot, test_size=0.2, random_state=42)
        _train_with_gradient_descent(X_train, X_test, y_train, y_test)

        path = self.save_locally(filename)
        self.save_to_s3(path, model_name)
        print("Saved:", path)

    def _train_with_gradient_descent(X_train, X_test, y_train, y_test):
        # Initialize training setup for gradient descent
        y_ = tf.placeholder("float", [None, 2])
        cost = -tf.reduce_mean(y_ * tf.log(y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        init = tf.initialize_all_variables()
        self.sess.run(init)

        # Run gradient descent on parameters
        for i in range(100):
            average_cost = 0
            number_of_batches = int(len(X_train) / batch_size)
            for start, end in zip(range(0, len(X_train), batch_size), range(batch_size, len(X_train), batch_size)):
                self.sess.run(optimizer, feed_dict={X: X_train[start:end], y_: y_train[start:end]})
                # Compute average loss
                average_cost += self.sess.run(cost, feed_dict={X: X_train[start:end],
                                              y_: y_train[start:end]}) / number_of_batches
            print(self.sess.run(accuracy, feed_dict={self.X: X_test, y_: y_test}))

