import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from wine_quality.softmax_regression import softmax_regression
from boto.s3.connection import S3Connection


# Remove outliers
def _outliers(df, threshold, columns):
    for col in columns:
        mask = df[col] > float(threshold) * df[col].std() + df[col].mean()
        df.loc[mask==True, col] = np.nan  # noqa
        mean_property = df.loc[:, col].mean()
        df.loc[mask==True, col] = mean_property  # noqa
    return df


def _dense_to_one_hot(labels_dense, num_classes=2):
    # Convert class labels from scalars to one-hot vectors
    num_labels = len(labels_dense)
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense] = 1
    return labels_one_hot


class tf_model():

    def __init__(self):
        self.sess = tf.Session()
        x = tf.placeholder("float", [None, 10])

        with tf.variable_scope("softmax_regression"):
            self.X = tf.placeholder("float", [None, 10])
            self.y1, self.variables = softmax_regression(x)
            self.saver = tf.train.Saver(self.variables)

    def run_model(self, x1):
        return self.sess.run(self.y1, feed_dict={self.X: x1})

    def load_locally(self):
        # Load the data
        self.saver.restore(self.sess, "wine_quality/data/softmax_regression.ckpt")

    def save_locally(self, filename):
        path = self.saver.save(self.sess, os.path.join(os.path.dirname(__file__), filename))
        return path

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

    def train(self, training_df, learning_rate=0.001, batch_size=126, model_name="softmax_model"):
        column_list = training_df.columns.tolist()
        threshold = 5

        red_wine_cleaned = training_df.copy()
        red_wine_cleaned = _outliers(red_wine_cleaned, threshold, column_list[0:-1])

        # Bin the data
        bins = [3, 5, 6, 8]
        red_wine_cleaned['category'] = pd.cut(red_wine_cleaned.quality, bins, labels=['Bad', 'Average', 'Good'],
                                              include_lowest=True)

        # Only include 'Bad' and 'Good' categories
        red_wine_newcats = red_wine_cleaned[red_wine_cleaned['category'].isin(['Bad', 'Good'])].copy()

        bins = [3, 5, 8]
        red_wine_newcats['category'] = pd.cut(red_wine_newcats.quality,
                                              bins, labels=['Bad', 'Good'], include_lowest=True)

        y_red_wine = red_wine_newcats[['category']].get_values()

        # Removing fixed_acidity and quality
        X_red_wine = red_wine_newcats.iloc[:, 1:-2].get_values()

        y_red_wine_raveled = y_red_wine.ravel()
        y_red_wine_integers = [y.replace('Bad', '1') for y in y_red_wine_raveled]
        y_red_wine_integers = [y.replace('Good', '0') for y in y_red_wine_integers]
        y_red_wine_integers = [np.int(y) for y in y_red_wine_integers]

        y_one_hot = _dense_to_one_hot(y_red_wine_integers, num_classes=2)

        X_train, X_test, y_train, y_test = train_test_split(X_red_wine, y_one_hot, test_size=0.2, random_state=42)
        # model

        with tf.variable_scope("softmax_regression"):
            X = tf.placeholder("float", [None, 10])
            y, variables = softmax_regression(X)

        # train
        y_ = tf.placeholder("float", [None, 2])
        cost = -tf.reduce_mean(y_ * tf.log(y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        init = tf.initialize_all_variables()
        self.sess.run(init)
        for i in range(100):
            average_cost = 0
            number_of_batches = int(len(X_train) / batch_size)
            for start, end in zip(range(0, len(X_train), batch_size), range(batch_size, len(X_train), batch_size)):
                self.sess.run(optimizer, feed_dict={X: X_train[start:end], y_: y_train[start:end]})
                # Compute average loss
                average_cost += self.sess.run(cost, feed_dict={X: X_train[start:end],
                                              y_: y_train[start:end]}) / number_of_batches
            print(self.sess.run(accuracy, feed_dict={X: X_test, y_: y_test}))

        filename = "data/softmax_regression.ckpt"
        path = self.save_locally(filename)
        self.save_to_s3(path, model_name)
        print("Saved:", path)
