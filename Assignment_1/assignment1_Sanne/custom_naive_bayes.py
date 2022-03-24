import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from abs_custom_classifier_with_feature_generator import CustomClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

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
        self.prior = None
        self.word_likelihoods = None

    def fit(self, train_feats, train_labels):
        """ Fit training data for Naive Bayes classifier """
        total_training_sentences = train_feats.shape[0]
        total_by_class = np.bincount(train_labels)
        self.priors = [(item/total_training_sentences) for item in total_by_class] # calculating the priors

        self.unique_labels = np.unique(train_labels) # get unique labels
        features_by_label = [[] for item in self.unique_labels] # creates an empty list for each unique label
        for tweet, label in zip(train_feats, train_labels):
            features_by_label[label].append(tweet) # append all the feature vectors to corresponding label

        feature_sum_by_label = [[] for item in self.unique_labels]
        for label in self.unique_labels:
            feature_sum_by_label[label] = np.sum(features_by_label[label], axis=0, initial=1) # initial=1 is alpha smoothing, accumulating the word counts per label

        word_likelihoods = []
        for emotion in feature_sum_by_label:
            word_likelihoods_per_emotion = [(item/total_by_class[0]) for item in emotion] # divide word counts by total number of words for each class
            word_likelihoods.append(word_likelihoods_per_emotion)
        self.word_likelihoods = np.array(word_likelihoods) # convert to numpy array

        return self

    def predict(self, test_feats):
        """ Predict classes with provided test features """
        predictions = []
        for i, test_tweet in enumerate(test_feats):
            word_exists_index = np.flatnonzero(test_tweet) # get word indexes non-zero count
            likelihoods_of_tweet_by_label = [1 for item in self.unique_labels]
            for label in self.unique_labels:
                for index in word_exists_index:
                    likelihoods_of_tweet_by_label[label] *= (self.word_likelihoods[label][index] ** test_tweet[index]) # calculate likelihood of the tweet

            conditional_probabilities = [(likelihoods_of_tweet_by_label[0] * self.priors[0]),(likelihoods_of_tweet_by_label[1] * self.priors[1]),(likelihoods_of_tweet_by_label[2] * self.priors[2]),(likelihoods_of_tweet_by_label[3] * self.priors[3])] # calculating conditional probabilties per class
            prediction = np.argmax(np.array(conditional_probabilities)) # get the label of the tweet with the highest conditonal probability
            predictions.append(prediction)

        return predictions
