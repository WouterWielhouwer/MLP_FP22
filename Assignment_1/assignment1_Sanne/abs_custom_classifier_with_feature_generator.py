import abc
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

"""
Implement a classifier with required functions:

get_features: feature vector for each sample (1-hot, n-hot encodings or etc.)
fit(train_features, train_labels): to train the classifier
predict(test_features): to predict test labels 
"""

class CustomClassifier(abc.ABC):
    def __init__(self):
        self.counter = None

    def get_features(self, text_list, ngram=1):
        """ Get word count features per sentences as a 2D numpy array """
        if self.counter == None: # check if there counter already exists
            self.counter = CountVectorizer()
            encoded_tweet = self.counter.fit_transform(text_list) # if counter doesn't exist make countvectorizer and fit and transform
        else:
            encoded_tweet = self.counter.transform(text_list) # if counter already exist use transform
        features_array = encoded_tweet.toarray()

        return features_array

    @abc.abstractmethod
    def fit(self, train_features, train_labels):
        pass

    @abc.abstractmethod
    def predict(self, test_features):
        pass


