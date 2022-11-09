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

random.seed(5)
train_set = random.sample(range(0, 68), 50)
test_set = list(set(train_set) ^ set(range(0, 68)))
test_sub = []

annots = loadmat('pose.mat')
data = annots["pose"]

def llnorm(sample, mu,sigma):
    sample_centre = sample - mu

    ll = -0.5 * np.linalg.det(sigma) - 0.5*np.linalg.multi_dot([sample_centre.transpose(), np.linalg.inv(sigma), sample_centre])
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
            row = np.append(row,i)
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
            img_mp = skimage.measure.block_reduce(img, (3,3), np.max)
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
        unlabelled_data.append(data[i,:-1])
        labels.append(data[i,-1])
    return np.array(unlabelled_data),np.array(labels)

def center_values(data):
    """
    Centers the data around origin considering that each row is a sample
    :param data: data to be centered
    :return: centered data
    """
    data_cent = np.zeros(np.shape(data)[0])
    center_function = lambda x: x - x.mean()
    for i in range(np.shape(data)[1]):
        col = center_function(data[:,i])
        data_cent = np.column_stack((data_cent,col))
        # print("Added Column")
    return np.array(data_cent[:,1:])

def data_mean(data):
    """
    Returns the mean across all samples, considering that each row of the data is a single sample
    :param data: data to be centered
    :return: centered data
    """
    mu = []
    mu_func = lambda x: x.mean()
    for i in range(np.shape(data)[1]):
        mu_col = mu_func(data[:,i])
        mu.append(mu_col)
    return np.array(mu)

def PCA(data, num_components = 50,scree_plot = False, reg=0.01):
    """
    Each row of the data is a sample.
    :param data: data matrix
    :param num_components:
    :param scree_plot:
    :return: Reduced data matrix
    """

    # covariance matrix is the relation of each pixel with every other pixel for the data
    # for numpy.cov() each row is a variable and each column is a single observation
    cov_mat = np.cov(data.transpose())
    cov_mat = cov_mat + np.identity(len(cov_mat))*reg
    eig_val, eig_vec = np.linalg.eigh(cov_mat)

    # sorting eigenvalues and eigenvectors
    idx = eig_val.argsort()[::-1]
    eigenValues = eig_val[idx]
    eigenVectors = eig_vec[:, idx]

    if scree_plot == True:
        #Plotting eigenvalues to decide the number of components to pick
        eigenValues_percent = (eigenValues/sum(eigenValues))*100
        plt.ylim(0,max(eigenValues_percent))
        plt.ylabel("% of Eigenvalues")
        plt.xlabel("Components")
        plt.bar(list(range(0,len(eigenValues[:100]))),eigenValues_percent[:100])
        plt.show()

    eigenVectors_subset = eigenVectors[:, 0:num_components]

    data_reduced = np.dot(eigenVectors_subset.transpose(), data.transpose()).transpose()

    return data_reduced, eigenVectors_subset

def data_grouping(data,labels):
    """
    Groups the data according to its labels
    :param data: The complete data matrix with labels
    :param labels: The labels for which the data is to be grouped. Can also input the labels list created with create_data_set()
    :return: a list of the grouped data
    """

    data_groups = []
    def group_data(data, label=1):
        data_grouped=np.empty(len(data[0,:]))
        for i in range(len(data)):
            if data[i,-1]==label:
                data_grouped = np.vstack((data_grouped,data[i,:]))
        data_grouped = data_grouped[1:,:]
        return data_grouped

    for i in set(labels):
        data_groups.append(group_data(data, label=i))

    return data_groups

def calc_likelihood(sample,data_group, reg=0.01):
    cov_mat = np.cov(data_group.transpose())
    cov_mat = cov_mat + np.identity(len(cov_mat)) * reg
    # cov = ShrunkCovariance().fit(data_group)
    mean = data_mean(data_group)

    y = llnorm(sample,mean,cov_mat)
    # y = multivariate_normal.pdf(sample, mean=mean, cov=cov.covariance_, allow_singular=True)
    return y

def sample_likelihood(sample,data_groups):
    likelihood = []
    for data_group in data_groups:
        data_group,label = remove_label(data_group)
        likelihood.append((calc_likelihood(sample,data_group),label))

    return likelihood

train_dat, test_dat = create_pose_data_set()
data, labels = remove_label(train_dat)

data_cent = center_values(data)

#Calculating correlation matrix to better understand relation between pixels
corr_mat = np.corrcoef(data_cent.transpose())
# data_cent_mp = skimage.measure.block_reduce(data_cent, (3,3), np.max)
# corr_mat_mp = np.corrcoef(data_cent_mp.transpose())

data_reduced, eigenvectors = PCA(data=data_cent)

data_reduced_w_labels = np.column_stack((data_reduced,labels))
data_cent_w_labels = np.column_stack((data_cent,labels))

data_groups = data_grouping(train_dat, labels)

test_dat_wout_labels, test_dat_labels = remove_label(test_dat)

test_data_reduced = np.dot(eigenvectors.transpose(), test_dat_wout_labels.transpose()).transpose()

likelihood = sample_likelihood(test_dat_wout_labels[2,:],data_groups)
likelihood = np.array(likelihood)
likelihood = likelihood[likelihood[:, 0].argsort()]

print("Done!")

"""
Feature Selection is pending
Check correlation matrix and remove unneccesary features 
P values are another option 
Max pooling seems like another option
Code for it is in scratch 11

Normalization is also pending, all the features should lie between zero and one. 
"""