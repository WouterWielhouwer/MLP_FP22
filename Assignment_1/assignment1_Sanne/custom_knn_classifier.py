import numpy as np
import scipy
from abs_custom_classifier_with_feature_generator import CustomClassifier
from scipy.spatial.distance import cdist

"""
Implement a KNN classifier with required functions:

fit(train_features, train_labels): to train the classifier
predict(test_features): to predict test labels 
"""


class CustomKNN(CustomClassifier):
    def __init__(self, k=5, distance_metric='cosine'):
        """ """
        super().__init__()

        self.k = k
        self.train_feats = None
        self.train_labels = None
        self.distance_metric = distance_metric


    def fit(self, train_feats, train_labels):
        """ Fit training data for Naive Bayes classifier """

        self.train_feats = train_feats
        self.train_labels = np.array(train_labels)

        return self

    def predict(self, test_feats):
        """ Predict classes with provided test features """
        predictions = []
        distance_values = cdist(test_feats, self.train_feats, self.distance_metric) # get the distance values for each test_tweet versus each training_tweet
        sorted_closest_tt = np.argsort(distance_values) # sort all of the distance values, so the indexes of the lowest distances (highest similarity) will be on the left. !arg.sort() gives the indexes
        for tweet in sorted_closest_tt:
            k_closest_tweets = tweet[0:self.k] # makes a list with the indexes of the K closest tweets
            labels_n_neighbours = []
            for i in range(len(k_closest_tweets)):
                labels_n_neighbours.append(self.train_labels[k_closest_tweets[i]]) # get the labels of the indexes of the K closest tweets
            votes_per_class = np.bincount(labels_n_neighbours) # counts how many times each index is occuring
            max_voted_class = np.argmax(votes_per_class) # gets (the index of) the class with the most votes
            predictions.append(max_voted_class)

        return predictions
