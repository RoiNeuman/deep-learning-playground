import multiprocessing

from binary_classification.models.residual_dense_binary_classifier import residual_dense_binary_classifier
from data_preprocess.airline_passenger_satisfaction_dataset import load_airline_passenger_satisfaction_dataset
from utils.statistics import plot_confusion_matrix


def run():
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_airline_passenger_satisfaction_dataset(train_ratio=0.8,
                                                                                                 validation_ratio=0.1,
                                                                                                 test_ratio=0.1,
                                                                                                 max_feature_dims=None)

    # Create the model
    # model = dense_binary_classifier(input_dim=X_train.shape[1],
    #                                 hidden_layers_units=[X_train.shape[1] / 2] * 20,
    #                                 dropout=0.3,
    #                                 batch_norm=True)
    model = residual_dense_binary_classifier(input_dim=X_train.shape[1],
                                             hidden_layers_units=[X_train.shape[1]] * 10,
                                             dropout=0.2,
                                             batch_norm=True)
    model.summary()

    # Train the model
    model.fit(X_train, Y_train,
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
