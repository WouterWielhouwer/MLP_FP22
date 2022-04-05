import time
from preprocessing import *
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier

def train_test(train_feats, test_feats, train_labels):
    """ Trains with the training features and training labels, test on the test features and returns the predicted test labels"""
    cls = GaussianNB()
    cls.fit(train_feats, train_labels)
    predicted_test_labels = cls.predict(test_feats)

    return predicted_test_labels

def south_park():
    """"Reads the South Park file and returns the dataset"""
    south = pd.read_csv("South_Park/All-seasons.csv", quotechar='"')
    south.name = 'South Park'
    df = create_df(south, ["Character", "Line"])

    return (df, south.name)

def game_of_thrones():
    """""Reads the Game of Thrones file and returns the dataset"""
    got = pd.read_csv("Game_of_Thrones_Script/Game_of_Thrones_Script.csv", quotechar='"')
    got.name = 'Game Of Thrones'
    df = create_df(got, ["Name", "Sentence"])

    return (df, got.name)

def run(df, name, n_characters, n_gram):
    """"Runs the training and testing and evaluates the results"""
    df = pool_other(df, n_characters) # Make n classes in dataset df, were n is the number of characters, the rest will get the 'Other' class
    df.character = pd.Categorical(df.character) # Makes the characters categorical
    df = remove_other(df) # Removes the 'Other' class from the dataset

    train_data, test_data, train_labels, test_labels = split_train_test(df, 0.25) # Splits the dataset in training and testing
    train_feats, test_feats = generate_features(train_data, test_data, n_gram) # Generates features, were n-gram could be for example 2 (bigram)
    predicted_labels = train_test(train_feats, test_feats, train_labels)  # Training and testing
    accuracy = accuracy_score(test_labels, predicted_labels) # Calculating accuracy

    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(train_feats, train_labels)
    print("Baseline: ", dummy_clf.score(test_feats, test_labels))

    print(name, n_characters, n_gram, accuracy)
    f = open("scores.txt", "a")
    f.write("dataset: " + str(name) + '\n' + "number of characters:" + str(n_characters) + '\n' + 'ngrams:' + str(n_gram) + '\n' + 'accuracy: ' + str(accuracy) + "\n\n\n")
    f.close()

def main():
    start_time = time.time()
    n_characters = [3, 5, 7, 10]
    sp, south_name = south_park()
    got, got_name = game_of_thrones()
    for n in n_characters:
        df_copy_south = sp.copy(deep=True)
        df_copy_got = got.copy(deep=True)
        run(df_copy_south, south_name, n, 1)
        run(df_copy_south, south_name, n, 2)
        run(df_copy_south, south_name, n, 3)
        run(df_copy_got, got_name, n, 1)
        run(df_copy_got, got_name, n, 2)
        run(df_copy_got, got_name, n, 3)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
