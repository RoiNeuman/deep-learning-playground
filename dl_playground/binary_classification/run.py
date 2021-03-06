import multiprocessing

import numpy as np

from dl_playground.binary_classification.models.transformer_binary_classifier import transformer_binary_classifier
from dl_playground.data_preprocess.airline_passenger_satisfaction_dataset import \
    load_airline_passenger_satisfaction_dataset
from dl_playground.utils.statistics import plot_confusion_matrix


def run():
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_airline_passenger_satisfaction_dataset(train_ratio=0.7,
                                                                                                 validation_ratio=0.1,
                                                                                                 test_ratio=0.2,
                                                                                                 max_feature_dims=None)

    # Create the model
    input_dim = X_train.shape[1]
    embed_dim = 16
    # model = residual_dense_binary_classifier(input_dim=X_train.shape[1],
    #                                          hidden_layers_units=[i + 1 for i in range(X_train.shape[1], 0, -1)],
    #                                          dropout_rate=0.2,
    #                                          batch_norm=True)
    model = transformer_binary_classifier(input_dim=input_dim,
                                          attention_heads=[4] * 8,
                                          embed_dim=embed_dim,
                                          ff_dim=embed_dim)
    model.summary()

    # Expanding the data to the embedding dimension
    X_train = np.repeat(np.expand_dims(X_train, axis=2), embed_dim, axis=2)
    X_val = np.repeat(np.expand_dims(X_val, axis=2), embed_dim, axis=2)
    X_test = np.repeat(np.expand_dims(X_test, axis=2), embed_dim, axis=2)

    # Train the model
    model.fit(x=X_train,
              y=Y_train,
              validation_data=(X_val, Y_val),
              epochs=10,
              batch_size=256,
              use_multiprocessing=True,
              workers=multiprocessing.cpu_count())

    # Evaluate the model on the test set
    test_predictions = model.evaluate(X_test, Y_test)
    print("Loss = " + str(test_predictions[0]))
    print("Test Accuracy = " + str(test_predictions[1]))
    predictions = model.predict(X_test)
    # print("Test Accuracy = " + str(np.count_nonzero(np.around(test_predictions) == Y_test) / Y_test.shape[0]))

    # Plotting confusion matrix
    plot_confusion_matrix(Y_test, predictions)
