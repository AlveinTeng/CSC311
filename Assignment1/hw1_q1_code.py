import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def load_data(texts):
    x_label = []
    y_label = []
    for t in texts:
        (x, y) = t
        x_label.append(x)
        y_label.append(y)
    cv = CountVectorizer()

    X = cv.fit_transform(x_label)

    # set seed
    np.random.seed(10)

    # split first time
    X_train, X_test_1, Y_train, Y_test_1 = \
        train_test_split(X, y_label, test_size=0.3)

    # split second time
    X_vali, X_test, Y_vali, Y_test = train_test_split(X_test_1, Y_test_1, test_size=0.5)

    return X_train, X_vali, X_test, Y_train, Y_vali, Y_test


def select_knn_model(x_train, x_vali, y_train, y_vali, cosine, X_test, Y_test):
    ks, trains, valis = [], [], []
    best = -float("inf")
    best_model = None
    k = 1
    while k < 21:
        print("k={}".format(k))
        ks.append(k)
        if cosine:
            md = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        else:
            md = KNeighborsClassifier(n_neighbors=k)
        md.fit(x_train, y_train)
        train = md.score(x_train, y_train)
        trains.append(train)
        vali = md.score(x_vali, y_vali)
        valis.append(vali)
        print("     Training Accuracy:{}".format(train))
        print("     Validation Accuracy:{}".format(vali))

        if best <= vali:
            best = vali
            best_model = md
        k += 1
    print("Best validation accuracy:{}".format(best))
    print("Best accuracy on the test data: {}".format(best_model.score(X_test, Y_test)))
    plt.plot(ks, trains, label="training")
    plt.plot(ks, valis, label="validation")
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.title("Training and Validation Accuracy vs k")
    plt.legend()
    plt.show()

    return best_model


if __name__ == '__main__':
    data_set = []
    for r in open("data/clean_real.txt", "r").readlines():
        data_set.append((r, "real data"))

    for f in open("data/clean_fake.txt", "r").readlines():
        data_set.append((f, "fake data"))

    X_train, X_vali, X_test, Y_train, Y_vali, Y_test = load_data(data_set)

    b_model = select_knn_model(X_train, X_vali, Y_train,
                               Y_vali, False, X_test ,Y_test)
    b_model = select_knn_model(X_train, X_vali, Y_train,
                               Y_vali, True, X_test, Y_test)


