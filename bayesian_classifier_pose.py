import numpy as np
import scipy.linalg
from numpy import *
import pandas
import matplotlib.pyplot as plt
from scipy.io import loadmat
import random
from scipy.stats import multivariate_normal
import skimage.measure
import math
from sklearn.covariance import ShrunkCovariance
import time

random.seed(5)
train_set = random.sample(range(0, 68), 50)
test_set = list(set(train_set) ^ set(range(0, 68)))
test_sub = []

annots = loadmat('pose.mat')
data = annots["pose"]


def llnorm(sample, mu, sigma):
    sample_centre = sample - mu

    ll = -0.5 * np.linalg.det(sigma) - 0.5 * np.linalg.multi_dot(
        [sample_centre.transpose(), np.linalg.inv(sigma), sample_centre])
    return ll


def create_pose_data_set():
    """
    shuffles and returns train data and test data with each row corresponding to a flattened image.The last element of the row is the label.
    :return: (numpy array, numpy array)
    """

    train_dat = []
    for j in train_set:
        for i in range(0, 13):
            row = data[:, :, i, j].flatten()
            row = np.append(row, i)
            train_dat.append(row)
    random.shuffle(train_dat)

    test_dat = []
    for j in test_set:
        for i in range(0, 13):
            row = data[:, :, i, j].flatten()
            row = np.append(row, i)
            test_dat.append(row)
    random.shuffle(test_dat)

    return np.array(train_dat), np.array(test_dat)


def create_pose_data_set_mp():
    """
    shuffles and returns train data and test data with each row corresponding to a flattened image.The last element of the row is the label.
    :return: (numpy array, numpy array)
    """

    train_dat = []
    for j in train_set:
        for i in range(0, 13):
            img = data[:, :, i, j]
            img_mp = skimage.measure.block_reduce(img, (3, 3), np.max)
            row = img_mp.flatten()
            row = np.append(row, i)
            train_dat.append(row)
    random.shuffle(train_dat)

    test_dat = []
    for j in test_set:
        for i in range(0, 13):
            img = data[:, :, i, j]
            img_mp = skimage.measure.block_reduce(img, (3, 3), np.max)
            row = img_mp.flatten()
            row = np.append(row, i)
            test_dat.append(row)
    random.shuffle(test_dat)

    return np.array(train_dat), np.array(test_dat)


def remove_label(data):
    """
    Removes label from numpy array with each row corresponding to a flattened image
    :param train_dat: numpy array with labels
    :return: numpy array without labels, labels
    """
    unlabelled_data = []
    labels = []
    for i in range(np.shape(data)[0]):
        unlabelled_data.append(data[i, :-1])
        labels.append(data[i, -1])
    return np.array(unlabelled_data), np.array(labels)


def data_mean(data):
    """
    Returns the mean across all samples, considering that each row of the data is a single sample
    :param data: data to be centered
    :return: centered data
    """
    mu = []
    mu_func = lambda x: x.mean()
    for i in range(np.shape(data)[1]):
        mu_col = mu_func(data[:, i])
        mu.append(mu_col)
    return np.array(mu)


def data_grouping(data, labels):
    """
    Groups the data according to its labels
    :param data: The complete data matrix with labels
    :param labels: The labels for which the data is to be grouped. Can also input the labels list created with create_data_set()
    :return: a list of the grouped data
    """

    data_groups = []

    def group_data(data, label=1):
        data_grouped = np.empty(len(data[0, :]))
        for i in range(len(data)):
            if data[i, -1] == label:
                data_grouped = np.vstack((data_grouped, data[i, :]))
        data_grouped = data_grouped[1:, :]
        return data_grouped

    for i in set(labels):
        data_groups.append(group_data(data, label=i))

    return data_groups


def calc_likelihood(sample, data_group, reg=0.01):
    cov_mat = np.cov(data_group.transpose())
    cov_mat = cov_mat + np.identity(len(cov_mat)) * reg
    # cov = ShrunkCovariance().fit(data_group)
    mean = data_mean(data_group)

    y = llnorm(sample, mean, cov_mat)
    # y = multivariate_normal.pdf(sample, mean=mean, cov=cov.covariance_, allow_singular=True)
    return y


def sample_likelihood(sample, data_groups):
    likelihood = []
    for data_group in data_groups:
        data_group, label = remove_label(data_group)
        likelihood.append((calc_likelihood(sample, data_group), label[0]))

    return likelihood

start = time.time()

train_dat, test_dat = create_pose_data_set()
data, labels = remove_label(train_dat)
test_dat, test_labels = remove_label(test_dat)

data_groups = data_grouping(train_dat, labels)

count = 0
for i, test_row in enumerate(test_dat):
    likelihood = sample_likelihood(test_row, data_groups)
    likelihood.sort(key=lambda tup: tup[0], reverse=True)
    prediction = likelihood[0][1]
    print('Expected %d, Got %d.' % (test_labels[i], prediction))
    if test_labels[i] == prediction: count = count + 1

print(f'Accuracy is {count / np.shape(test_dat)[0]}')
end = time.time()
print(f"Time taken = {end-start} seconds")
print("Done!")


"""
Feature Selection is pending
Check correlation matrix and remove unneccesary features 
P values are another option 
Max pooling seems like another option
Code for it is in scratch 11

Normalization is also pending, all the features should lie between zero and one. 
"""