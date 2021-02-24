import pandas as pd
import matplotlib.pyplot as plt


def check_imbalance(df: pd.DataFrame, column: str):
    fig = plt.figure(figsize=(8, 5))
    df[column].value_counts(normalize=True).plot(kind='bar', color=['darkorange', 'steelblue'], alpha=0.9, rot=0)
    plt.title(f'{column.capitalize()} Indicator (0) and (1) in the Dataset')
    plt.show()
