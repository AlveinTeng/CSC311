from q2.check_grad import check_grad
from q2.utils import *
from q2.logistic import *

# from check_grad import check_grad
# from utils import *
# from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    # Uncomment if train set
    # hyperparameters = {
    #     "learning_rate": 0.1,
    #     "weight_regularization": 0,
    #     "num_iterations": 1170
    # }

    # Uncomment if small train set
    # hyperparameters = {
    #     "learning_rate": 0.001,
    #     "weight_regularization": 0,
    #     "num_iterations": 1487
    # }

    weights = 0.001 * np.random.randn(M + 1, 1)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    ts, ce_trs, ce_vs, ce_ts = [], [], [], []
    e_trs, e_vs, e_ts = [], [], []

    low_t, l_ce_tr, l_e_tr = 0, 0, 0
    l_ce_v, l_e_v, l_ce_te, l_e_te = 1487, 0, 0, 0

    for t in range(hyperparameters["num_iterations"]):
        ts.append(t)

        f, df, y_tr = logistic(weights, train_inputs, train_targets, hyperparameters)
        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")
        ce_trs.append(f)
        ce_tr, cr_tr = evaluate(train_targets, y_tr)
        e_trs.append(1 - cr_tr)

        weights = weights - hyperparameters['learning_rate'] * df

        y_v = logistic_predict(weights, valid_inputs)
        y_te = logistic_predict(weights, test_inputs)

        ce_v, cr_v = evaluate(valid_targets, y_v)
        ce_t, cr_te = evaluate(test_targets, y_te)

        ce_vs.append(ce_v)
        ce_ts.append(ce_t)

        e_vs.append(1 - cr_v)
        e_ts.append(1 - cr_te)

        if ce_v < l_ce_v:
            l_ce_tr, l_e_tr = ce_tr, 1 - cr_tr
            l_ce_v, l_e_v = ce_v, 1 - cr_v
            l_ce_te, l_e_te = ce_t, 1 - cr_te
            low_t = t

    print("t: {}".format(low_t + 1))
    print("Train: cross entropy: {}, classification error: {}".format(l_ce_tr, l_e_tr * 100))
    print("Vali: cross entropy: {}, classification error: {}".format(l_ce_v, l_e_v * 100))
    print("Test: cross entropy: {}, classification error: {}".format(l_ce_te, l_e_te * 100))

    fig, graph = plt.subplots()
    graph.plot(ts, ce_trs, label="train")
    graph.plot(ts, ce_vs, label="validation")

    # Uncomment if train set
    graph.set(xlabel='number of iterations', ylabel='Cross Entropy',
              title="Q2.2")

    # Uncomment if small train set
    # graph.set(xlabel='number of iterations', ylabel='Cross Entropy',
    #           title="Q2.2 small")

    graph.grid()
    graph.legend()

    # Uncomment if train set
    fig.savefig("ce.png")

    # Uncomment if small train set
    # fig.savefig("ce_small.png")

    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_pen_logistic_regression():
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Implement the function that automatically evaluates different     #
    # penalty and re-runs penalized logistic regression 5 times.        #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.001,
        "weight_regularization": 0,
        "num_iterations": 2000
    }

    lambds = [0, 0.001, 0.01, 0.1, 1.0]

    for lambd in lambds:
        hyperparameters["weight_regularization"] = lambd
        avg_ce_trs, avg_ce_vs, avg_e_trs, avg_e_vs, avg_ce_tes, avg_e_ts = [], [], [], [], [], []
        ts = [t for t in range(hyperparameters["num_iterations"])]
        i = 0
        while i < 5:
            ce_trs, e_trs, ce_vs, e_vs, ce_ts, e_ts = [], [], [], [], [], []
            weights = 0.001 * np.random.randn(M + 1, 1)
            for t in range(hyperparameters["num_iterations"]):
                f, df, y_tr = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
                if np.isnan(f) or np.isinf(f):
                    raise ValueError("nan/inf error")
                ce_trs.append(f)
                ce_tr, cr_tr = evaluate(train_targets, y_tr)
                e_trs.append(1 - cr_tr)

                weights = weights - hyperparameters['learning_rate'] * df

                y_v = logistic_predict(weights, valid_inputs)
                y_te = logistic_predict(weights, test_inputs)

                ce_v, cr_v = evaluate(valid_targets, y_v)
                ce_t, cr_te = evaluate(test_targets, y_te)

                ce_vs.append(ce_v)
                ce_ts.append(ce_t)

                e_vs.append(1 - cr_v)
                e_ts.append(1 - cr_te)

                avg_ce_trs.append(ce_trs)
                avg_ce_vs.append(ce_vs)
                avg_ce_tes.append(ce_ts)
                avg_e_trs.append(e_trs)
                avg_e_vs.append(e_vs)
                avg_e_ts.append(e_ts)
            i += 1

        avg_ce_trs = np.mean(avg_ce_trs, axis=0)
        avg_ce_vs = np.mean(avg_ce_vs, axis=0)

        avg_e_tr = np.average(np.mean(avg_e_trs, axis=0))
        avg_e_v = np.average(np.mean(avg_e_vs, axis=0))
        avg_e_t = np.average(np.mean(avg_e_ts, axis=0))
        avg_ce_tr = np.average(avg_ce_trs)
        avg_ce_v = np.average(avg_ce_vs)
        avg_ce_te = np.average(np.mean(avg_ce_tes, axis=0))

        print("lambd={}".format(lambd))
        print("Train: cross entropy: {}, classification error: {}".format(avg_ce_tr, avg_e_tr * 100))
        print("Vali: cross entropy: {}, classification error: {}".format(avg_ce_v, avg_e_v * 100))
        print("Test: cross entropy: {}, classification rate: {}".format(avg_ce_te, (1 - avg_e_t) * 100))

        fig, graph = plt.subplots()
        graph.plot(ts, avg_ce_trs, label="train")
        graph.plot(ts, avg_ce_vs, label="validation")
        # Uncomment if train set
        # graph.set(xlabel='number of iterations', ylabel='Cross Entropy',
        #           title="Q2.3 lambda={}".format(lambd))

        # Uncomment if small train set
        graph.set(xlabel='number of iterations', ylabel='Cross Entropy',
                  title="Q2.3 small lambda={}".format(lambd))

        graph.grid()
        graph.legend()

        # Uncomment if train set
        # fig.savefig("ce_{}.png".format(lambds.index(lambd)))

        # Uncomment if small train set
        fig.savefig("ce_small_{}.png".format(lambds.index(lambd)))

        # plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    # run_logistic_regression()
    run_pen_logistic_regression()
