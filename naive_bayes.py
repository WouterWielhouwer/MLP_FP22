from preprocessing import *
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


def fit(train_feats, train_labels):
    """ Fit training data for Naive Bayes classifier """
    total_training_sentences = train_feats.shape[0]
    total_by_class = np.bincount(train_labels)
    priors = [(item / total_training_sentences) for item in total_by_class]  # calculating the priors

    unique_labels = np.unique(train_labels)  # get unique labels
    features_by_label = [[] for item in unique_labels]  # creates an empty list for each unique label
    for tweet, label in zip(train_feats, train_labels):
        features_by_label[label].append(tweet)  # append all the feature vectors to corresponding label

    feature_sum_by_label = [[] for item in unique_labels]
    for label in unique_labels:
        feature_sum_by_label[label] = np.sum(features_by_label[label], axis=0,
                                             initial=1)  # initial=1 is alpha smoothing, accumulating the word counts per label

    word_likelihoods = []
    for emotion in feature_sum_by_label:
        word_likelihoods_per_emotion = [(item / total_by_class[0]) for item in
                                        emotion]  # divide word counts by total number of words for each class
        word_likelihoods.append(word_likelihoods_per_emotion)
    word_likelihoods = np.array(word_likelihoods)  # convert to numpy array

    return unique_labels, word_likelihoods, priors


def predict(unique_labels, word_likelihoods, priors, test_feats):
    """ Predict classes with provided test features """
    predictions = []
    for i, test_tweet in enumerate(test_feats):
        word_exists_index = np.flatnonzero(test_tweet)  # get word indexes non-zero count
        likelihoods_of_tweet_by_label = [1 for item in unique_labels]
        for label in unique_labels:
            for index in word_exists_index:
                likelihoods_of_tweet_by_label[label] *= (word_likelihoods[label][index] ** test_tweet[
                    index])  # calculate likelihood of the tweet

        conditional_probabilities = [(likelihoods_of_tweet_by_label[0] * priors[0]),
                                     (likelihoods_of_tweet_by_label[1] * priors[1]),
                                     (likelihoods_of_tweet_by_label[2] * priors[2]), (
                                                 likelihoods_of_tweet_by_label[3] * priors[
                                             3])]  # calculating conditional probabilties per class
        prediction = np.argmax(
            np.array(conditional_probabilities))  # get the label of the tweet with the highest conditonal probability
        predictions.append(prediction)

    return predictions


def main():
    south = pd.read_csv("South_Park/All-seasons.csv")
    south.name = 'South Park'

    df = create_df(south, ["Character", "Line"])
    df = pool_other(df, 3)

    train_data, test_data, train_labels, test_labels = split_train_test(df, 0.25)
    train_feats, test_feats = generate_features(train_data, test_data, 2)

    unique_labels, word_likelihoods, priors = fit(train_feats, train_labels)
    predictions = predict(unique_labels, word_likelihoods, priors, test_feats)
    print(predictions)
