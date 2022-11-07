import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import multivariate_normal
import random

random.seed(5)
train_set = random.sample(range(0, 68), 50)
test_set = list(set(train_set) ^ set(range(0, 68)))
test_sub = []

annots = loadmat('pose.mat')
data = annots["pose"]


def create_data():
    test_dat = []
    for j in test_set:
        for i in range(0, 13):
            test_dat.append([data[:, :, i, j].flatten(), i])

    random.shuffle(test_dat)

    train_dat = []
    for j in train_set:
        for i in range(0, 13):
            train_dat.append([data[:, :, i, j].flatten(), i])

    random.shuffle(train_dat)

    return train_dat, test_dat

def remove_label(train_dat):
    data = []
    for i in range(np.shape(train_dat)[0]):
        data.append(train_dat[i][0])

    return np.array(data)

def center_values(data):
    data_cent = np.zeros(np.shape(data)[0])
    center_function = lambda x: x - x.mean()
    for i in range(np.shape(data)[1]):
        col = center_function(data[:,i])
        data_cent = np.column_stack((data_cent,col))
        # print("Added Column")
    return data_cent[:,1:]

def mean(data):
    mu = []
    mu_func = lambda x: x.mean()
    for i in range(np.shape(data)[1]):
        mu_col = mu_func(data[:,i])
        mu.append(mu_col)
    return np.array(mu)


train_dat, test_dat = create_data()
data = remove_label(train_dat)
data_cent = center_values(data)
mean = mean(data)
# for numpy.cov() each row is a variable and each column is a single observation
# we have 1920 variables and 650 observations for each variable thus transpose is required before calculating covariance matrix
cov_mat = np.cov(data.transpose())
y = multivariate_normal.pdf(data[5,:], mean=mean, cov=cov_mat, allow_singular=True)
#Stuck since the covariance matrix is a singular matrix



print("Done!")