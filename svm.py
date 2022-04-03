import pandas as pd
import time
from sklearn.svm import SVC
from preprocessing import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def train_test(X_train_vec, X_test_vec, X_labels):
    cls = SVC(kernel='linear')
    cls.fit(X_train_vec, X_labels)
    predicted_test_labels = cls.predict(X_test_vec)

    return predicted_test_labels

def south_park():
    south = pd.read_csv("South_Park/All-seasons.csv")
    south.name = 'South Park'
    df = create_df(south, ["Character", "Line"])
    return (df, south.name)

def game_of_thrones():
    got = pd.read_csv("Game_of_Thrones_Script/Game_of_Thrones_Script.csv")
    got.name = 'Game Of Thrones'
    df = create_df(got, ["Name", "Sentence"])

    return (df, got.name)

def run(df, name, n_characters, n_gram):
    df = pool_other(df, n_characters)
    df.character = pd.Categorical(df.character)
    df = remove_other(df)

    X_train, y_train, X_labels, y_labels = split_train_test(df, 0.25)

    X_train_vec, X_test_vec = generate_features(X_train, y_train, n_gram)

    predicted_test_labels = train_test(X_train_vec, X_test_vec, X_labels)

    accuracy = accuracy_score(y_labels, predicted_test_labels)

    matrix = confusion_matrix(y_labels, predicted_test_labels)
    accs = matrix.diagonal() / matrix.sum(axis=1)

    f = open("scores.txt", "a")
    f.write("dataset: " + str(name) + '\n' + "number of characters:" + str(n_characters) + '\n' + 'ngrams:' + str(n_gram) + '\n' + 'accuracy: ' + str(accuracy) + '\n' + str(accs) + "\n\n\n")
    f.close()

def main():
    start_time = time.time()
    n_characters = [3, 5, 7, 10]
    sp, south_name = south_park()
    got, got_name = game_of_thrones()
    for n in n_characters:
        run(sp, south_name, n, 1)
        #run(sp, south_name, n, 2)
        #run(sp, south_name, n, 3)
        run(got, got_name, n, 1)
        #run(got, got_name, n, 2)
        #run(got, got_name, n, 3)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
