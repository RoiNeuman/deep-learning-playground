import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from binary_classification.models.residual_dense_binary_classifier import residual_dense_binary_classifier
from data_preprocess.airline_passenger_satisfaction_dataset import load_airline_passenger_satisfaction_dataset


def run():
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_airline_passenger_satisfaction_dataset(train_ratio=0.75,
                                                                                                 validation_ratio=0.15,
                                                                                                 test_ratio=0.1)

    # Create the model
    # model = dense_binary_classifier(input_dim=X_train.shape[1],
    #                                 hidden_layers_units=[X_train.shape[1] / 2] * 20,
    #                                 dropout=0.3,
    #                                 batch_norm=True)
    model = residual_dense_binary_classifier(input_dim=X_train.shape[1],
                                             hidden_layers_units=[X_train.shape[1]] * 20,
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

    cm = confusion_matrix(y_true=Y_test, y_pred=np.around(predictions))
    plt.imshow(cm, cmap=plt.cm.Blues, origin='lower')
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([0, 1], [0, 1])
    plt.yticks([0, 1], [0, 1])
    for (j, i), label in np.ndenumerate(cm):
        plt.text(i, j, label, ha='center', va='center')
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.show()
