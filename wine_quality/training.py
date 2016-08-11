import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import wine_quality.model as model

# Load the data

# Remove outliers
def _outliers(df, threshold, columns):
    for col in columns:
        mask = df[col] > float(threshold)*df[col].std()+df[col].mean()
        df.loc[mask == True,col] = np.nan
        mean_property = df.loc[:,col].mean()
        df.loc[mask == True,col] = mean_property
    return df

def _dense_to_one_hot(labels_dense, num_classes=2):
    # Convert class labels from scalars to one-hot vectors
    num_labels = len(labels_dense)
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense] = 1
    return labels_one_hot

def train_model(training_df, learning_rate=0.001, batch_size=126):
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
    red_wine_newcats['category'] = pd.cut(red_wine_newcats.quality, bins, labels=['Bad', 'Good'], include_lowest=True)


    y_red_wine = red_wine_newcats[['category']].get_values()
    # Removing fixed_acidity and quality
    X_red_wine = red_wine_newcats.iloc[:,1:-2].get_values()

    y_red_wine_raveled = y_red_wine.ravel()
    y_red_wine_integers = [y.replace('Bad', '1') for y in y_red_wine_raveled]
    y_red_wine_integers = [y.replace('Good', '0') for y in y_red_wine_integers]
    y_red_wine_integers = [np.int(y) for y in y_red_wine_integers]


    y_one_hot = _dense_to_one_hot(y_red_wine_integers, num_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(X_red_wine, y_one_hot, test_size=0.2, random_state=42)
    # model

    with tf.variable_scope("softmax_regression"):
        X = tf.placeholder("float", [None, 10])
        y, variables = model.softmax_regression(X)

    # train
    y_ = tf.placeholder("float", [None, 2])
    cost = -tf.reduce_mean(y_*tf.log(y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver(variables)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    log_list = []  # List to store logging of model progress
    for i in range(100):
        average_cost = 0
        number_of_batches = int(len(X_train) / batch_size)
        for start, end in zip(range(0, len(X_train), batch_size), range(batch_size, len(X_train), batch_size)):
            sess.run(optimizer, feed_dict={X: X_train[start:end], y_: y_train[start:end]})
            # Compute average loss
            average_cost += sess.run(cost, feed_dict={X: X_train[start:end], y_: y_train[start:end]}) / number_of_batches
        print("Epoch:", '%04d' % (i + 1), "cost=", "{:.9f}".format(average_cost))
        log_cost = "Epoch: {:d}, cost= {:.9f}".format(i + 1, average_cost)
        # print(log_cost)
        log_list.append(log_cost)

    print("Finished optimization!")
    log_list.append("Finished optimization!")

    print("Accuracy: {0}".format(sess.run(accuracy, feed_dict={X: X_test, y_: y_test})))
    log_accuracy = "Accuracy: {0}".format(sess.run(accuracy, feed_dict={X: X_test, y_: y_test}))
    # print(log_accuracy)
    log_list.append(log_accuracy)

    path = saver.save(sess, os.path.join(os.path.dirname(__file__), "data/softmax_regression.ckpt"))
    print("Saved:", path)
    log_list.append("Saved: "+path)

    print("")
    print(log_list)

    # return log_list

