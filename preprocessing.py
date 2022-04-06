import pandas as pd
import os
import re
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def create_df(dataset, columnnames):
    """
    :param dataset: pandas dataframe of the dataset
    :param columnnames: names of the columns with relevant information. Column for character should be named first
    :return: pandas dataframe transformed to be used in further operations
    """

    df = dataset[columnnames]
    df = df.rename(
        columns={columnnames[0]: "character", columnnames[1]: "line"})

    return df


def pool_other(df, n=7):
    """
    :param df: pandas dataframe with names and lines
    :param n: number of main characters to be left
    :return: transformed pandas dataframe with n+1 characters of which one is labelled "other"
    """

    names = df.character.value_counts().index.tolist()[:n]
    df["character"] = df["character"].apply(
        lambda x: "other" if x not in names else x)
    df.character = pd.Categorical(df.character)

    return df


def remove_other(df):
    """
    :param df: dataframe of characters and lines
    :return: dataframe without characters named other
    """
    return df[df.character != "other"]


def get_mapping(df):
    """
    :param df: pandas dataframe with names and lines
    :return: mapping of integer value associated with label
    """
    return dict(enumerate(df['character'].cat.categories))


def tok_norm_sent(line):
    """
    :param line: string of text
    :return tokenised and normalized string of text. A period is also added before the first word to indicate start of sentence.
    """
    tokens = word_tokenize(line.lower())
    norm = ["."]
    for token in tokens:
        if token in "!?":
            token = "."
            norm.append(token)
        elif token not in ",()\"`":
            norm.append(token)
    return norm


def dummy(doc):
    return doc


def split_train_test(df, split=.25):
    """
    :param df: pandas dataframe
    :param split: split threshold (float between 0 and 1)
    :return: train data, test data, training labels and test labels
    """

    X = df.line
    y = df.character.cat.codes

    return train_test_split(X, y, test_size=split, random_state=69)


def generate_features(train_data, test_data, ngram_range):
    """
    :param train_data
    :param test_data
    :param ngram_range: size of ngrams used in vectorization
    :return: vectorized train features and test features
    """

    vectorizer = CountVectorizer(tokenizer=tok_norm_sent, preprocessor=dummy)
    vectorizer.ngram_range = (ngram_range, ngram_range)

    train_features = vectorizer.fit_transform(train_data).toarray()
    test_features = vectorizer.transform(test_data).toarray()

    return train_features, test_features


