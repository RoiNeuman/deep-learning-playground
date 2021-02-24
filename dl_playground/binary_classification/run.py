from binary_classification.data_preprocess import load_heart_dataset
from binary_classification.models.dense_binary_classifier import dense_binary_classifier


def run():
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_heart_dataset(train_ratio=0.6,
                                                                        validation_ratio=0.2,
                                                                        test_ratio=0.2)

    # Create the model
    model = dense_binary_classifier(input_dim=X_train.shape[1],
                                    hidden_layers_units=[16, 8, 8],
                                    dropout=0.1,
                                    batch_norm=True)
    model.summary()

    # Train the model
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=350, batch_size=16)

    # Evaluate the model on the test set
    test_predictions = model.evaluate(X_test, Y_test)
    print("Loss = " + str(test_predictions[0]))
    print("Test Accuracy = " + str(test_predictions[1]))
    # test_predictions = model.predict(X_test)
    # print("Test Accuracy = " + str(np.count_nonzero(np.around(test_predictions) == Y_test) / Y_test.shape[0]))
