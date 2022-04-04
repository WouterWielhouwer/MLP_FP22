from preprocessing import *
import pandas as pd
import numpy as np
from sklearn import neighbors


def fit_pred(df, name, n_char, size=10000, n_gram=2, other=False):
    df = pool_other(df, n_char)
    df.character = pd.Categorical(df.character)

    if not other:
        df = remove_other(df)
        other_str = "without"
    else:
        other_str = "with"

    df = df[:size]

    X_train, X_test, y_train, y_test = split_train_test(df, 0.25)

    X_train_vec, X_test_vec = generate_features(X_train, X_test, n_gram)

    best_score = 0
    best_k = None
    for k in range(1, 71, 2):
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_vec, y_train)
        score = knn.score(X_test_vec, y_test)
        if score > best_score:
            best_score = score
            best_k = k

    print("best score for %s with %s characters, %s class \"Other\", sample size of %s and ngram=%s: %s \n k = %s" % (name,
                                                                                                       str(n_char),
                                                                                                       other_str,
                                                                                                       str(size),
                                                                                                       str(n_gram),
                                                                                                       str(best_score),
                                                                                                       str(best_k)))


def main():
    south = pd.read_csv("South_Park/All-seasons.csv")

    df_south = create_df(south, ["Character", "Line"])

    got = pd.read_csv("Game_of_Thrones_Script/Game_of_Thrones_Script.csv")

    df_got = create_df(got, ["Name", "Sentence"])

    for i in [3, 5, 7, 10]:
        try:
            for x in [True, False]:
                fit_pred(df_south, "South Park", i, size=3000, other=x)

        except Exception as e:
            print(e)

    for i in [3, 5, 7, 10]:
        try:
            for x in [True, False]:
                fit_pred(df_got, "Game of Thrones", i, size=3000, other=x)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
