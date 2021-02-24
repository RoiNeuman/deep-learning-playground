import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_airline_passenger_satisfaction_dataset(train_ratio: float = 0.75,
                                                validation_ratio: float = 0.15,
                                                test_ratio: float = 0.10):
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

    # Label encoding of categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

    # Separating features and labels
    features = ['Type_of_Travel', 'Inflight_wifi_service', 'Online_boarding', 'Seat_comfort', 'Flight_Distance',
                'Inflight_entertainment', 'On-board_service', 'Leg_room_service', 'Cleanliness', 'Checkin_service',
                'Inflight_service', 'Baggage_handling']
    target = ['satisfaction']
    X = df[features]
    y = df[target].to_numpy()

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
