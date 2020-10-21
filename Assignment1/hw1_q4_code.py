import numpy as np
from matplotlib import pyplot as plt


data_train = {'X': np.genfromtxt('data_train_X.csv', delimiter=','),
              't': np.genfromtxt('data_train_y.csv', delimiter=',')}
data_test = {'X': np.genfromtxt('data_test_X.csv', delimiter=','),
             't': np.genfromtxt('data_test_y.csv', delimiter=',')}


def get_lambd_seq(num):
    lambd_seq = []
    interval = (0.005 - 0.00005) / num

    for i in range(0, 50):
        lambd = interval * i + 0.00005
        lambd_seq.append(lambd)
    return lambd_seq


def predict(data, model):

    predictions = np.dot(data[1], model)
    return predictions


def shuffle_data(data):
    (t, X) = data

    rnd = list(zip(t, X))
    np.random.seed(100)
    np.random.shuffle(rnd)
    t, X = zip(*rnd)

    data_shf = (np.asarray(t), np.asarray(X))
    return data_shf


def split_data(data, num_folds, fold):
    (t, X) = data

    t_lst = np.split(t, num_folds)
    X_lst = np.array_split(X, num_folds)
    data_fold = (np.asarray(t_lst.pop(fold)), np.asarray(X_lst.pop(fold)))

    rest_t = []
    rest_X = []

    while len(t_lst) != 0:
        new_X = X_lst.pop(0)
        rest_X.extend(new_X.tolist())
        new_t = t_lst.pop(0)
        rest_t.extend(new_t.tolist())

    data_rest = (np.asarray(rest_t), np.asarray(rest_X))

    return data_fold, data_rest


def train_model(data, lambd):
    (t, X) = data

    N = t.size

    I = np.identity(X.shape[1])

    model = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + N * lambd * I), X.T), t)
    return model


def loss(data, model):
    (t, X) = data

    y = predict(data, model)
    N = t.size

    total = 0
    for i in range(0, N):
        total += (y[i] - t[i]) ** 2
    return total / (2 * N)


def cross_validation(data, num_folds, lambd_seq):
    cv_error = []
    data = shuffle_data(data)
    for i in range(0, len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0
        for fold in range(0, num_folds):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)

        cv_error.append(cv_loss_lmd / num_folds)
    return cv_error


if __name__ == '__main__':
    train_data = (data_train.get('t'), data_train.get('X'))
    test_data = (data_test.get('t'), data_test.get('X'))

    train_errors = []
    test_errors = []
    lambd_seq = get_lambd_seq(49)

    for lambd in lambd_seq:
        print("lambda: {}".format(lambd))
        tr_model = train_model(train_data, lambd)
        train_error = loss(train_data, tr_model)
        train_errors.append(train_error)

        test_error = loss(test_data, tr_model)
        test_errors.append(test_error)

        print("     train error: {}".format(train_error))
        print("     test error: {}".format(test_error))

    cv_errors_5 = cross_validation(train_data, 5, lambd_seq)

    cv_errors_10 = cross_validation(train_data, 10, lambd_seq)

    print("5-fold cross validation lambda: {}".format(lambd_seq[cv_errors_5.index(min(cv_errors_5))]))
    print("10-fold cross validation lambda: {}".format(lambd_seq[cv_errors_10.index(min(cv_errors_10))]))

    plt1 = plt.figure(1)
    plt.plot(lambd_seq, train_errors, label="train")
    plt.plot(lambd_seq, test_errors, label="test")
    plt.plot(lambd_seq, cv_errors_5, label="cv num_folds = 5", color="green")
    plt.plot(lambd_seq, cv_errors_10, label="cv num_folds = 10", color="red")
    plt.legend()
    plt.show()
    plt.savefig('Question_4')
