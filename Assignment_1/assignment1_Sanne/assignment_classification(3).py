import os
import numpy as np
import nltk
import string
from sklearn import metrics
from custom_knn_classifier import CustomKNN
from custom_naive_bayes import CustomNaiveBayes
from sklearn_svm_classifier import SVMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

##################################################################
####################### DATASET FUNCTIONS ########################
##################################################################

def read_dataset(folder, split):
    print('***** Reading the dataset *****')
    text_file = open(os.path.join(folder, f'{split}_text.txt'), encoding='utf-8')
    labels_file = open(os.path.join(folder, f'{split}_labels.txt'), encoding='utf-8')

    texts = [t.strip() for t in text_file]
    labels = [int(l.strip()) for l in labels_file]

    assert len(texts) == len(labels), 'Text and label files should have same number of lines..'
    print(f'Number of samples: {len(texts)}')

    return texts, labels


def preprocess_dataset(text_list):
    """
    Return the list of sentences after preprocessing. Example:
    >>> preprocess_dataset(['the quick brown fox #HASTAG-1234 @USER-XYZ'])
    ['the quick brown fox']
    """
    preprocessed_text_list = []
    for sentence in text_list:
        tokenized_list = nltk.word_tokenize(sentence) # tokenize each word in the sentence
        sentence_tn = ""
        for item in tokenized_list:
            sentence_tn = sentence_tn + ' ' + item
            if item == 'user':
                tokenized_list.remove(item) # remove all the 'user' strings in the sentence
        sentence_wp = " ".join([item for item in tokenized_list if item not in string.punctuation]) # removing punctuation including @ and #
        preprocessed_text_list.append(sentence_wp)

    return preprocessed_text_list


def get_label_mapping(folder):
    print('***** Reading the mapping file for labels *****')
    mapping = open(os.path.join(folder, 'mapping.txt'), encoding='utf-8')
    id2label = {int(m.strip().split('\t')[0]): m.strip().split('\t')[1] for m in mapping}
    print(id2label)

    return id2label


def max_voting_prediction(classifier_list, test_features, weights):
    """
    Implement a Voting classifier with hard voting.
    To get predicted labels for each sample use:
    >>> cls.predict(test_features=[[1,2,0], [3,5,0]])
    ['1', '0']
    Check soft voting implementation to see the structure.
    >>> max_voting_prediction(classifier_list=[cls1, cls2, cl3], test_features=[[1,2,0], [3,5,0]], weights=[1.,1.,1.])
    ['1', '0']
    """
    predicted_test_labels = None

    return predicted_test_labels


##################################################################
####################### EVALUATION METRICS #######################
##################################################################


def evaluate(true_labels, predicted_labels, label_names):
    """
    Print accuracy, precision, recall and f1 metrics for each classes and macro average.
    >>> evaluate(true_labels=[1,0,3,2,0], predicted_labels=[1,3,2,2,0], label_names=['A', 'B', 'C', 'D'])
    accuracy: 0.6
    precision: [1. , 1. , 0.5, 0. ]
    recall: [0.5, 1. , 1. , 0. ]
    f1: [0.66666667, 1. , 0.66666667, 0.]

    macro avg:
    precision: 0.625
    recall: 0.625
    f1: 0.583
    """

    confusion_matrix = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels) # making a confusion matrix for evaluation

    tp = np.diag(confusion_matrix) # get the true positives
    fp = np.sum(confusion_matrix, axis=0) - tp # get the false positives
    fn = np.sum(confusion_matrix, axis=1) - tp # get the false negatives
    tn = []

    # get the true negatives
    for i in range(4):
        x = np.delete(confusion_matrix, i, 0)
        x = np.delete(x, i, 1)
        tn.append(sum(sum(x)))

    accuracy = sum(tp + tn)/sum(tp + fp + fn + tn) # calculates accuracy
    precisions, recalls = tp/(tp + fp), tp/(tp + fn) # calculates precisions and recalls
    f1_measures = (2 * (precisions * recalls))/(precisions + recalls) # calculates f1_scores
    precision, recall, f1_measure = sum(precisions)/len(precisions), sum(recalls)/len(recalls), sum(f1_measures)/len(f1_measures)

    print('***** Evaluation *****')
    print(f'  Accuracy: {accuracy}')
    print(" |-----------|-----------|-----------|-----------|")
    print(" |%-11s|%-11s|%-11s|%-11s|" % ("Class", "Precision", "Recall", "F1-measure"))
    print(" |-----------|-----------|-----------|-----------|")
    for i, cls in enumerate(label_names):
        print(" |%-11s|%-11f|%-11f|%-11f|" % (cls, precisions[i], recalls[i], f1_measures[i]))
    print(" |-----------|-----------|-----------|-----------|")
    print(" |%-11s|%-11f|%-11f|%-11f|" % ('MACRO-AVG', precision, recall, f1_measure))
    print(" |-----------|-----------|-----------|-----------|")
    return f1_measure


def train_test(classifier='naive_bayes'):
    # Read train and test data and generate tweet list together with label list
    train_data, train_labels = read_dataset('tweet_classification_dataset', 'train')
    test_data, test_labels = read_dataset('tweet_classification_dataset', 'test')

    label_mapping = get_label_mapping('tweet_classification_dataset')
    # Preprocess train and test data
    #train_data = preprocess_dataset(train_data)
    #test_data = preprocess_dataset(test_data)

    # Create a your custom classifier
    if classifier == 'svm':
        cls = SVMClassifier()
    elif classifier == 'naive_bayes':
        cls = CustomNaiveBayes()
    elif classifier == 'knn':
        cls = CustomKNN(k=5, distance_metric='cosine')

    # Generate features from train and test data
    # features: word count features per sentences as a 2D numpy array
    train_feats = cls.get_features(train_data)
    test_feats = cls.get_features(test_data)

    # Train classifier
    cls.fit(train_feats, train_labels)

    # Predict labels for test data by using trained classifier and features of the test data
    predicted_test_labels = cls.predict(test_feats)

    # Evaluate the classifier by comparing predicted test labels and true test labels
    evaluate(test_labels, predicted_test_labels, label_mapping.values())


def cross_validate(n_fold=5, classifier='svm'):
    """
    Implement N-fold (n_fold) cross-validation by randomly splitting taining data/features into N-fold
    Store f1-mesure scores in a list for result of each fold and return this list
    Check main() for using required functions
    >>> cross_validate(n_fold=3, classifier='svm')
    [0.5, 0.4, 0.6]
    """
    scores = []
    train_data, train_labels = read_dataset('tweet_classification_dataset', 'train') # get training data and labels
    cls = SVMClassifier()
    train_feats = cls.get_features(train_data) # get training features
    kf = KFold(n_splits=n_fold) # Split dataset into k consecutive folds (without shuffling)
    train_labels = np.array(train_labels)

    for train_index, test_index in kf.split(train_feats):
        features_train, features_test = train_feats[train_index], train_feats[test_index]
        labels_train, labels_test = train_labels[train_index], train_labels[test_index]
        cls.fit(features_train, labels_train) # Train classifier
        predicted_test_labels = cls.predict(features_test) # Predict labels for test data by using trained classifier and features of the test data
        f_score = f1_score(labels_test, predicted_test_labels, average='macro') # Calculating f-score
        scores.append(f_score)
    print(f'Average F1-measure for {n_fold}-fold: {np.mean(np.array(scores))}')

    return np.mean(np.array(scores))


def main():
    train_test()
    #cross_validate(n_fold=5, classifier='svm')

if __name__ == "__main__":
    main()
