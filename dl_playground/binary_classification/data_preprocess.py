import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_heart_dataset(train_ratio: float = 0.75, validation_ratio: float = 0.15, test_ratio: float = 0.10):
    df = pd.read_csv('./datasets/heart.csv')

    y = df.target.values
    x_data = df.drop(['target'], axis=1)

    # Normalize the dataset
    X = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

    # Split the dataset to train, validation, test parts
    X_train, X_test, Y_train, Y_test = train_test_split(X, y,
                                                        test_size=test_ratio,
                                                        # random_state=1
                                                        )
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                      test_size=validation_ratio / (train_ratio + validation_ratio),
                                                      # random_state=1
                                                      )

    return np.array(X_train), np.array(Y_train).reshape((Y_train.shape[0], 1)), np.array(X_val), np.array(
        Y_val).reshape((Y_val.shape[0], 1)), np.array(X_test), np.array(Y_test).reshape((Y_test.shape[0], 1))
