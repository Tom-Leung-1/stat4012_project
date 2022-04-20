# reference https://www.kaggle.com/code/skmuhammadasif/performance-comparison-with-raw-balanced-data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

seed = 4012
test_size = 0.2

def resampling(X, y, seed):
    smt = SMOTETomek(sampling_strategy="auto", random_state=seed)
    sme = SMOTEENN(sampling_strategy="auto", random_state=seed)
    X_smt, y_smt = smt.fit_resample(x_train, y_train)
    X_sme, y_sme = sme.fit_resample(x_train, y_train)
    return X_smt, y_smt, X_sme, y_sme

def cleaning(df):
    df = df.dropna()
    df = df[df["DebtRatio"] < 1]
    return df.iloc[:, 1:], df.iloc[:, 0]

def data_visualization(df):
    print("NULL values count:")
    print(df.isnull().sum())
    n = len(df)
    labels = list(df["SeriousDlqin2yrs"].value_counts())
    print("\nBinary Class comparison:")
    for idx, val in enumerate(labels):
        print(idx, ":", val/n)
    sns.countplot(x="SeriousDlqin2yrs", data=df)
    plt.title("Default Comparison")
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("cs-training.csv").iloc[:, 1:]
    data_visualization(df)
    X, y = cleaning(df)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_smt, y_smt, X_sme, y_sme = resampling(x_train, y_train, seed)

