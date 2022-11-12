from __future__ import division, print_function

import math

import numpy as np
import cvxopt
from scipy.io import loadmat

from sklearn import datasets


# from mlfromscratch.utils import Plot


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


def rbf_kernel(gamma, **kwargs):
    def f(x1, x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)

    return f

def polynomial_kernel(power, coef, **kwargs):
    def f(x1, x2):
        return (np.inner(x1, x2) + coef)**power
    return f

def remove_label(train_dat):
    data = []
    labels = []
    for i in range(np.shape(train_dat)[0]):
        data.append(train_dat[i][0])
        labels.append(train_dat[i][1])
    return np.array(data),np.array(labels)

def classifier(sample, lagr_multi, support_vector_labels, support_vectors, intercept, alpha):
    coef = 1
    power = 2
    def kernel(x1, x2):
        return (np.inner(x1, x2) + coef) ** power

    prediction = 0
    for i in range(len(lagr_multi)):
        prediction += lagr_multi[i] * support_vector_labels[
            i] * kernel(support_vectors[i], sample)
    prediction += intercept

    return np.sign(alpha*prediction)


def main():
    # data = datasets.load_iris()
    # X = normalize(data.data[data.target != 0])
    # y = data.target[data.target != 0]
    # y[y == 1] = -1
    # y[y == 2] = 1

    annots = loadmat('data.mat')
    face_data = annots["face"]
    neutral_face = []
    expression_face = []
    for i in range(0, int(np.shape(face_data)[2] / 3)):
        neutral_face.append([face_data[:, :, 3 * i].flatten(), 1])
        expression_face.append([face_data[:, :, (3 * i) + 1].flatten(), -1])

    data = np.vstack((neutral_face, expression_face))

    X = np.empty(len(data[0][0]))
    Y = []
    for i in range(len(data)):
        X = np.vstack((X, data[i][0]))
        Y.append(data[i][1])

    X = X[1:, :]
    y = np.array(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    # clf = SupportVectorMachine(kernel=polynomial_kernel, power=2, coef=1)
    weights = np.full(len(y_train), 1 / len(y_train))
    alpha = []
    lagr_multi = []
    support_vectors_labels = []
    support_vectors = []
    intercept = []

    for i in range(4):
        X = np.empty(np.shape(X_train)[1])
        for id, row in enumerate(X_train):
            X_row = np.multiply(row, weights[id])
            X = np.vstack((X, X_row))
        X = X[1:, :]

        clf = SupportVectorMachine(kernel=polynomial_kernel, power=2, coef=1)
        lagr_multi_temp, support_vectors_labels_temp, support_vectors_temp, intercept_temp = clf.fit(X, y_train)

        lagr_multi.append(lagr_multi_temp)
        support_vectors_labels.append(support_vectors_labels_temp)
        support_vectors.append(support_vectors_temp)
        intercept.append(intercept_temp)

        y_pred = []
        classification = []
        misclassified_weights = []


        for idx, sample in enumerate(X_train):
            prediction = classifier(sample, lagr_multi, support_vectors_labels, support_vectors, intercept, alpha=1)
            # Is this correct?? Should alpha be applied before or after the sign function
            # prediction = alpha * prediction

            y_pred.append(prediction)
            if y[idx] == prediction:
                classification.append(True)
            else:
                classification.append(False)

        for id, bool in enumerate(classification):
            if bool == False:
                misclassified_weights.append(weights[id])

        epsilon = sum(misclassified_weights)
        alpha.append(0.5 * math.log((1 - epsilon) / epsilon))

        # Updating weights
        for id, bool in enumerate(classification):
            if bool == False:
                weights[id] = weights[id] / (2 * epsilon)
            else:
                weights[id] = weights[id] / (2 * (1 - epsilon))


    for i in range(4):
        for idx, sample in enumerate(X_train):
            prediction = classifier(sample, lagr_multi, support_vectors_labels, support_vectors, intercept, alpha=1)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print("Accuracy:", accuracy)



    # Reduce dimension to two using PCA and plot the results
    # Plot().plot_in_2d(X_test, y_pred, title="Support Vector Machine", accuracy=accuracy)


# Hide cvxopt output
cvxopt.solvers.options['show_progress'] = False


class SupportVectorMachine(object):
    """The Support Vector Machine classifier.
    Uses cvxopt to solve the quadratic optimization problem.
    Parameters:
    -----------
    C: float
        Penalty term.
    kernel: function
        Kernel function. Can be either polynomial, rbf or linear.
    power: int
        The degree of the polynomial kernel. Will be ignored by the other
        kernel functions.
    gamma: float
        Used in the rbf kernel function.
    coef: float
        Bias term used in the polynomial kernel function.
    """

    def __init__(self, C=1, kernel=rbf_kernel, power=4, gamma=None, coef=4):
        self.C = C
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None

    def fit(self, X, y):

        n_samples, n_features = np.shape(X)

        # Set gamma to 1/n_features by default
        if not self.gamma:
            self.gamma = 1 / n_features

        # Initialize kernel method with parameters
        self.kernel = self.kernel(
            power=self.power,
            gamma=self.gamma,
            coef=self.coef)

        # Calculate kernel matrix
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])

        # Define the quadratic optimization problem
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if not self.C:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # Solve the quadratic optimization problem using cvxopt
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        lagr_mult = np.ravel(minimization['x'])

        # Extract support vectors
        # Get indexes of non-zero lagr. multipiers
        idx = lagr_mult > 1e-7
        # Get the corresponding lagr. multipliers
        self.lagr_multipliers = lagr_mult[idx]
        # Get the samples that will act as support vectors
        self.support_vectors = X[idx]
        # Get the corresponding labels
        self.support_vector_labels = y[idx]

        # Calculate intercept with first support vector
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[
                i] * self.kernel(self.support_vectors[i], self.support_vectors[0])

        return self.lagr_multipliers, self.support_vector_labels, self.support_vectors, self.intercept

    def predict(self, X):
        y_pred = []
        # Iterate through list of samples and make predictions
        for sample in X:
            prediction = 0
            # Determine the label of the sample by the support vectors
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[
                    i] * self.kernel(self.support_vectors[i], sample)
            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)

        print("done")


if __name__ == "__main__":
    main()