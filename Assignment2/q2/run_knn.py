from q2.l2_distance import l2_distance
from q2.utils import *

# from l2_distance import l2_distance
# from utils import *
#
import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # TODO:                                                             #
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################
    ks, cr_vs, cr_ts = [], [], []
    k = 1
    while k < 10:
        ks.append(k)
        print("k: {}".format(k))

        y_v = knn(k, train_inputs, train_targets, valid_inputs)
        y_t = knn(k, train_inputs, train_targets, test_inputs)
        acc_v, acc_t = 0, 0
        for i in range(y_v.size):
            if y_v[i] == valid_targets[i]:
                acc_v = acc_v + 1
        for j in range(y_t.size):
            if y_t[j] == test_targets[j]:
                acc_t = acc_t + 1
        cr_v, cr_t = acc_v / y_v.size, acc_t / y_t.size

        cr_vs.append(cr_v)
        cr_ts.append(cr_t)

        print("classification rate on validation set: {}".format(cr_v))
        print("classification rate on test set: {}".format(cr_t))

        k += 2

    plt.plot(ks, cr_vs, label="validation")
    # Uncomment if run classification rate on test set
    plt.plot(ks, cr_ts, label="test")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    run_knn()
