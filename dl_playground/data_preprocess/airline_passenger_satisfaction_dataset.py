import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_airline_passenger_satisfaction_dataset(train_ratio: float = 0.75,
                                                validation_ratio: float = 0.15,
                                                test_ratio: float = 0.10,
                                                max_feature_dims: int or None = None):
    df = pd.read_csv('./datasets/airline_passenger_satisfaction.csv')

    # Drop unnecessary columns
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop('id', axis=1)

    # Replace spaces in the column names with underscore
    df.columns = [c.replace(' ', '_') for c in df.columns]

    # Encoding string columns
    df['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1}, inplace=True)

    # Checking the nature of data set: balanced or imbalanced?
    # check_imbalance(df, 'satisfaction')

    # Imputing missing value with mean
    total_missing = df.isnull().sum().sort_values(ascending=False)
    keys_with_missing = total_missing[total_missing != 0].keys()
    df[keys_with_missing] = df[keys_with_missing].fillna(df[keys_with_missing].mean())

    # Replace NaN with mode for categorical variables
    for key in df.select_dtypes(include=['object']).columns:
        df[key] = df[key].fillna(df[key].mode()[0])

    # Encoding of categorical variables
    for col in df.select_dtypes(include=['object']).columns:

        # Creating One-Hot encoding for the different categories
        dummies = pd.get_dummies(df[col])

        # Applying PCA on the One-Hot vectors for:
        # 1. Dimensionality reduction
        # 2. Capture linear correlations between the different categories (Their amount in the dataset)
        pca_components = dummies.shape[1]
        if max_feature_dims is not None:
            pca_components = min(max_feature_dims, dummies.shape[1])
        pca = PCA(n_components=pca_components)
        dummies = pd.DataFrame(pca.fit_transform(dummies),
                               columns=[f'{col}_pca_{i + 1}' for i in range(pca_components)])

        # Adding the new encoded columns the removing the original column
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)

    # Separating features and labels
    target = ['satisfaction']
    y = df[target].to_numpy()
    df = df.drop(target, axis=1)
    X = df

    # Normalize Features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

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
