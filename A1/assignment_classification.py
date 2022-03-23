import os
import numpy as np
from sklearn import metrics
from custom_knn_classifier import CustomKNN
from custom_naive_bayes import CustomNaiveBayes
from sklearn_svm_classifier import SVMClassifier


##################################################################
####################### DATASET FUNCTIONS ########################
##################################################################

def read_dataset(folder, split):
    print('***** Reading the dataset *****')
    text_file = open(os.path.join(folder, f'{split}_text.txt'),
                     encoding='utf-8')
    labels_file = open(os.path.join(folder, f'{split}_labels.txt'),
                       encoding='utf-8')

    texts = [t.strip() for t in text_file]
    labels = [int(l.strip()) for l in labels_file]

    assert len(texts) == len(
        labels), 'Text and label files should have same number of lines..'
    print(f'Number of samples: {len(texts)}')

    return texts, labels


def preprocess_dataset(text_list):
    """
    Return the list of sentences after preprocessing. Example:
    >>> preprocess_dataset(['the quick brown fox #HASTAG-1234 @USER-XYZ'])
    ['the quick brown fox']
    """
    preprocessed_text_list = None
    return preprocessed_text_list


def get_label_mapping(folder):
    print('***** Reading the mapping file for labels *****')
    mapping = open(os.path.join(folder, 'mapping.txt'), encoding='utf-8')
    id2label = {int(m.strip().split('\t')[0]): m.strip().split('\t')[1] for m
                in mapping}
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
    accuracy: 0.6
    precision: [1. , 1. , 0.5, 0. ]
    recall: [0.5, 1. , 1. , 0. ]
    f1: [0.66666667, 1. , 0.66666667, 0.]

    macro avg:
    precision: 0.625
    recall: 0.625
    f1: 0.583
    """

    nptrue = np.array(true_labels)
    nppred = np.array(predicted_labels)
    accuracy = (nptrue == nppred).sum() / len(true_labels)
    precisions, recalls, f1_measures = [], [], []

    # TODO: account for zero divisions
    for i, label in enumerate(label_names):
        TP = ((nppred == i) & (nptrue == i)).sum()
        FP = ((nppred == i) & (nptrue != i)).sum()
        FN = ((nppred != i) & (nptrue == i)).sum()
        precisions.append(TP / (TP + FP))
        recalls.append(TP / (TP + FN))
        f1_measures.append(2 * TP / (2 * TP + (FP + FN)))

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    f1_measure = sum(f1_measures) / len(f1_measures)

    print('***** Evaluation *****')
    print(f'  Accuracy: {accuracy}')
    print(" |-----------|-----------|-----------|-----------|")
    print(" |%-11s|%-11s|%-11s|%-11s|" % (
        "Class", "Precision", "Recall", "F1-measure"))
    print(" |-----------|-----------|-----------|-----------|")
    for i, cls in enumerate(label_names):
        print(" |%-11s|%-11f|%-11f|%-11f|" % (
            cls, precisions[i], recalls[i], f1_measures[i]))
    print(" |-----------|-----------|-----------|-----------|")
    print(" |%-11s|%-11f|%-11f|%-11f|" % (
        'MACRO-AVG', precision, recall, f1_measure))
    print(" |-----------|-----------|-----------|-----------|")
    return f1_measure


def train_test(classifier='svm'):
    # Read train and test data and generate tweet list together with label list
    train_data, train_labels = read_dataset('tweet_classification_dataset',
                                            'train')
    test_data, test_labels = read_dataset('tweet_classification_dataset',
                                          'test')

    # Preprocess train and test data
    # train_data = preprocess_dataset(train_data)
    # test_data = preprocess_dataset(test_data)

    # Create a your custom classifier
    if classifier == 'svm':
        cls = SVMClassifier()
    elif classifier == 'naive_bayes':
        cls = CustomNaiveBayes()
    elif classifier == 'knn':
        cls = CustomKNN(k=5, distance_metric='cosine')

    # Generate features from train and test data
    # features: word count features per sentences as a 2D numpy array
    train_feats = cls.get_features(train_data, ngram=2)
    test_feats = cls.get_features(test_data)

    # Read label index-to-name mapping
    label_mapping = get_label_mapping('tweet_classification_dataset')

    # Train classifier
    cls.fit(train_feats, train_labels)

    # Predict labels for test data by using trained classifier and features of the test data
    predicted_test_labels = cls.predict(test_feats)

    # Evaluate the classifier by comparing predicted test labels and true test labels
    evaluate(test_labels, predicted_test_labels, label_mapping.values())


def cross_validate(n_fold=10, classifier='svm'):
    """
    Implement N-fold (n_fold) cross-validation by randomly splitting taining data/features into N-fold
    Store f1-mesure scores in a list for result of each fold and return this list
    Check main() for using required functions
    >>> cross_validate(n_fold=3, classifier='svm')
    [0.5, 0.4, 0.6]
    """
    scores = []
    print(f'Average F1-measure for {n_fold}-fold: {np.mean(np.array(scores))}')
    return np.mean(np.array(scores))


def main():
    train_test()


if __name__ == "__main__":
    main()
