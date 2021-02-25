import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def check_imbalance(df: pd.DataFrame, column: str):
    fig = plt.figure(figsize=(8, 5))
    df[column].value_counts(normalize=True).plot(kind='bar', color=['darkorange', 'steelblue'], alpha=0.9, rot=0)
    plt.title(f'{column.capitalize()} Indicator (0) and (1) in the Dataset')
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, predictions: np.ndarray):
    cm = confusion_matrix(y_true=y_true, y_pred=np.around(predictions))
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
