import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from abs_custom_classifier_with_feature_generator import CustomClassifier

"""
Implement a Naive Bayes classifier with required functions:

fit(train_features, train_labels): to train the classifier
predict(test_features): to predict test labels 
"""


class CustomNaiveBayes(CustomClassifier):

    def __init__(self, alpha=1.0):
        """ """
        super().__init__()

        self.alpha = alpha
        self.priors = None
        self.word_likelihoods = None

    def fit(self, train_feats, train_labels):
        """ Fit training data for Naive Bayes classifier """

        n_tweets = train_feats.shape[0]
        n_tweets_by_class = np.bincount(train_labels)
        self.priors = n_tweets_by_class/n_tweets

        unique_labels = np.unique(train_labels)
        features_by_label = [[] for c in unique_labels]
        for tweet, label in zip(train_feats, train_labels):
            features_by_label[label].append(tweet)

        feature_sum_by_label = [[] for c in unique_labels]
        for label in unique_labels:
            feature_sum_by_label[label] = np.sum(features_by_label[label],
                                                 axis=0)+1
        return self

    def predict(self, test_feats):
        """ Predict classes with provided test features """

        predictions = []
        return predictions
