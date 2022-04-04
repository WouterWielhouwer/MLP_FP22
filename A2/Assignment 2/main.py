import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering

from utils import plot_tsne, plot_confusion_matrix, show_image_mnist, \
    get_kernel, convolve_data
from kmeans import KMeansClusterer


def calc_purity(labels, predictions, n_clusters):  # Task 7
    count = 0
    for i in range(n_clusters):
        pred_i = predictions[np.nonzero(labels == i)]

        count += np.count_nonzero(pred_i == np.argmax(np.bincount(pred_i)))

    return count / len(labels)


def ag_clustering(data, labels):
    linkages = ['ward', 'complete', 'average', 'single']
    scores = {}
    # Instantiate agglomerative clustering
    for link in linkages:
        ag_clust = AgglomerativeClustering(n_clusters=10, linkage=link)
        ag_labels = ag_clust.fit_predict(data)
        score = adjusted_rand_score(labels, ag_labels)

        scores[link] = score

    return max(scores, key=scores.get)


# in hindsight I could have made these functions a lot less redundant and less confusion but alas, I didn't.
def conv_ag_clustering(kernel_names, data, labels):
    maxscore = 0
    bestkernel = ""
    for name in kernel_names:
        kernel = get_kernel(name)
        convolved = convolve_data(data, kernel)
        conv_ag_labels = AgglomerativeClustering(n_clusters=10,
                                                 linkage="ward").fit_predict(
            convolved)
        score = adjusted_rand_score(labels, conv_ag_labels)
        print("score for hierarchical clustering with %s kernel: %s" % (
            name, score))

        if score > maxscore:
            maxscore = score
            bestkernel = name

    return maxscore, bestkernel


# This is what I should have written in the first place
def c_a_c_predict(name, data):
    kernel = get_kernel(name)
    convolved = convolve_data(data, kernel)
    conv_ag_labels = AgglomerativeClustering(n_clusters=10,
                                             linkage="ward").fit_predict(
        convolved)
    return conv_ag_labels


def c_a_c_matrix(name, data, labels):
    kernel = get_kernel(name)
    convolved = convolve_data(data, kernel)
    conv_ag_labels = AgglomerativeClustering(n_clusters=10,
                                             linkage="ward").fit_predict(
        convolved)
    plot_confusion_matrix(labels, conv_ag_labels)


def c_a_c_sorted(kernel_names, data, labels):
    max_scores = [0] * 10
    max_names = [""] * 10

    for name in kernel_names:
        kernel = get_kernel(name)
        convolved = convolve_data(data, kernel)
        conv_ag_labels = AgglomerativeClustering(n_clusters=10,
                                                 linkage="ward").fit_predict(
            convolved)

        for i in range(10):
            pred_i = conv_ag_labels[np.nonzero(labels == i)]

            score = np.count_nonzero(
                pred_i == np.argmax(np.bincount(pred_i))) / len(pred_i)
            if score > max_scores[i]:
                max_scores[i] = score
                max_names[i] = name

    return max_scores, max_names


def main():
    # Loading in the data
    data = np.load("data/data.npy")
    labels = np.load("data/labels.npy")

    best_link = ag_clustering(data, labels)

    ag_labels = AgglomerativeClustering(n_clusters=10,
                                        linkage=best_link).fit_predict(data)

    print(adjusted_rand_score(labels, ag_labels))

    # plot_confusion_matrix(labels, ag_labels)

    # plot_tsne(data, ag_labels, labels)

    # scores per sobel filter size
    # conv_ag_clustering(
    #    ["sobel_0_3x3", "sobel_0_5x5", "sobel_0_7x7", "sobel_0_9x9"], data,
    #    labels)

    # different scores for   7x7 sobel filter rotations
    # names = []
    # for i in range(0, 91, 10):
    #    names.append("sobel_" + str(i) + "_7x7")

    # score, name = conv_ag_clustering(names, data, labels)

    # c_a_c_matrix("sobel_0_7x7", data, labels)

    # print("the best scoring kernel is %s with a score of %s" % (name, score))

    # max_scores, max_names = c_a_c_sorted(names, data, labels)

    # for i in range(10):
    #    print("best rotation for digit %s = %s  score = %s" % (i, max_names[i], max_scores[i]))

    # different scores for gaussian filter
    # names = []
    # for i in range(1, 20):
    #    names.append("gaussian_blur-"+ str(i) + "x" + str(i))

    # score, name = conv_ag_clustering(names, data, labels)

    # print("the best scoring gaussian filter is %s with a score of %s" % (name, score))

    # max_scores, max_names = c_a_c_sorted(names, data, labels)

    # for i in range(10):
    #    print("best filter for digit %s = %s  score = %s" % (i, max_names[i], max_scores[i]))

    # generating several predictions for purity calculations
    # print("purity score for agglomerative clustering without filters: %s" % calc_purity(labels, ag_labels, 10))
    # for name in ["sobel_0_7x7", "moldy_frikandel", "gaussian_blur_2x2"]:
    #    print("purity score for agglomerative clustering with %s: %s" % (name, calc_purity(labels, c_a_c_predict(name, data), 10)))


if __name__ == "__main__":
    main()
