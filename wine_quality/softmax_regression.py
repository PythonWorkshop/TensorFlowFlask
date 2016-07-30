import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import wine_quality.model as model

# Load the data
red_wine = pd.read_csv('data/winequality-red.csv', sep=';')

# Remove outliers
def _outliers(df, threshold, columns):
    for col in columns:
        mask = df[col] > float(threshold)*df[col].std()+df[col].mean()
        df.loc[mask == True,col] = np.nan
        mean_property = df.loc[:,col].mean()
        df.loc[mask == True,col] = mean_property
    return df


# Convert labels into one-hot vector format
def _dense_to_one_hot(labels_dense, num_classes=2):
    # Convert class labels from scalars to one-hot vectors
    num_labels = len(labels_dense)
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense] = 1
    return labels_one_hot


# Function to convert string categories into integers for making one-hot vectors
def _make_integer_labels(label_array):
    labels_raveled = label_array.ravel()
    label_series = pd.Series(labels_raveled)
    unique_labels = label_series.unique()
    labels_to_replace = np.sort(unique_labels)
    replace_with = np.arange(len(labels_to_replace)).tolist()
    integer_labels = label_series.replace(to_replace=labels_to_replace, value=replace_with).tolist()

    return integer_labels


# Function to train the model
def train_model():

    column_list = red_wine.columns.tolist()
    threshold = 5  # Set threshold to 5 standard deviations

    # Remove outliers
    red_wine_cleaned = red_wine.copy()
    red_wine_cleaned = _outliers(red_wine_cleaned, threshold, column_list[0:-1])

    # Bin the data into three separate categories
    bins = [3, 5, 6, 8]
    red_wine_cleaned['category'] = pd.cut(red_wine_cleaned.quality, bins, labels=['Bad', 'Average', 'Good'],
                                          include_lowest=True)

    # Only include 'Bad' and 'Good' categories
    red_wine_newcats = red_wine_cleaned[red_wine_cleaned['category'].isin(['Bad', 'Good'])].copy()

    bins = [3, 5, 8]
    red_wine_newcats['category'] = pd.cut(red_wine_newcats.quality, bins, labels=['Bad', 'Good'], include_lowest=True)

    # Extract categories column and save in an array for the labels
    y_red_wine = red_wine_newcats[['category']].get_values()
    print(pd.Series(y_red_wine.ravel()).unique())

    # Extract features and save to an array. Removing fixed_acidity and quality
    X_red_wine = red_wine_newcats.iloc[:,1:-2].get_values()

    # Convert string categories into integers that can be used to make one-hot vectors
    y_red_wine_integers = _make_integer_labels(y_red_wine)

    # Create one-hot vector array for labels
    y_one_hot = _dense_to_one_hot(y_red_wine_integers, num_classes=2)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_red_wine, y_one_hot, test_size=0.2, random_state=42)

    # Define model parameters
    learning_rate = 0.001
    batch_size = 126

    # Create placeholders and variables for input data and model
    with tf.variable_scope("softmax_regression"):
        X = tf.placeholder("float", [None, 10])
        y, variables = model.softmax_regression(X)

    y_ = tf.placeholder("float", [None, 2])

    # Define the cost and optimization functions
    cost = -tf.reduce_mean(y_*tf.log(y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Start the TensorFlow session and train the model
    saver = tf.train.Saver(variables)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    # Loop through each epoch
    for i in range(100):
        average_cost = 0
        number_of_batches = int(len(X_train) / batch_size)

        # Loop through each batch
        for start, end in zip(range(0, len(X_train), batch_size), range(batch_size, len(X_train), batch_size)):
            sess.run(optimizer, feed_dict={X: X_train[start:end], y_: y_train[start:end]})
            # Compute average loss
            average_cost += sess.run(cost, feed_dict={X: X_train[start:end], y_: y_train[start:end]}) / number_of_batches
        print(sess.run(accuracy, feed_dict={X: X_test, y_: y_test}))

        # Save the model
        path = saver.save(sess, os.path.join(os.path.dirname(__file__), "data/softmax_regression.ckpt"))
        print("Saved:", path)

train_model()
