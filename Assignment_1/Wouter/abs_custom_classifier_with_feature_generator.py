import abc
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy
import json

"""
Implement a classifier with required functions:

get_features: feature vector for each sample (1-hot, n-hot encodings or etc.)
fit(train_features, train_labels): to train the classifier
predict(test_features): to predict test labels 
"""


class CustomClassifier(abc.ABC):
    def __init__(self):
        pass

    def get_features(self, text_list, ngram=1):
        """ Get word count features per sentences as a 2D numpy array """
        try:
            features_array = self.counter.transform(text_list).toarray()
        except AttributeError:
            self.counter = CountVectorizer()
            self.counter.ngram_range = (ngram, ngram)
            features_array = self.counter.fit_transform(text_list).toarray()
        return features_array

    @abc.abstractmethod
    def fit(self, train_features, train_labels):
        pass

    @abc.abstractmethod
    def predict(self, test_features):
        pass
