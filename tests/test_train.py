from wine_quality.tf_model import tf_model
import pandas as pd


def test_train():
    model = tf_model()
    dataframe = pd.read_csv('wine_quality/data/winequality-red.csv', sep=',')
    model.train(dataframe, learning_rate=0.5, batch_size=126, model_name='model_name')


if __name__ == '__main__':
    test_train()
