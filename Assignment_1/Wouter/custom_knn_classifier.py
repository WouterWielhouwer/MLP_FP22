import numpy as np
import scipy
from sklearn.feature_extraction.text import CountVectorizer

from abs_custom_classifier_with_feature_generator import CustomClassifier

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

        # Use provided function by replacing X and Y, to calculate distance between test feature(s) and train feature(s)
        # You can use the function either by giving two matrices (All test features, All train features)
        # or by passing a matrix and a vector: (A test feature, All train features)
        distance_values = scipy.spatial.distance.cdist()

        predictions = []
        return predictions
