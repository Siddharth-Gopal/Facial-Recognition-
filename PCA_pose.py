import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.io import loadmat
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

train_dat, test_dat = create_data()

def remove_label(train_dat):
    data = []
    labels = []
    for i in range(np.shape(train_dat)[0]):
        data.append(train_dat[i][0])
        labels.append(train_dat[i][1])
    return np.array(data),np.array(labels)

data, labels = remove_label(train_dat)

def center_values(data):
    data_cent = np.zeros(np.shape(data)[0])
    center_function = lambda x: x - x.mean()
    for i in range(np.shape(data)[1]):
        col = center_function(data[:,i])
        data_cent = np.column_stack((data_cent,col))
        # print("Added Column")
    return data_cent[:,1:]

# for numpy.cov() each row is a variable and each column is a single observation
# we have 1920 variables and 650 observations for each variable thus transpose is required
data_cent = center_values(data)

#covariance matrix is 1920*1920, it is thus the relation of each pixel with every other pixel for the the entire train data
cov_mat = np.cov(data_cent.transpose())
eig_val, eig_vec = np.linalg.eigh(cov_mat)

idx = eig_val.argsort()[::-1]
eigenValues = eig_val[idx]
eigenVectors = eig_vec[:,idx]

# #Plotting eigenvalues to decide the number of components to pick
# eigenValues_percent = (eigenValues/sum(eigenValues))*100
# plt.ylim(0,40)
# plt.bar(list(range(0,len(eigenValues[:100]))),eigenValues_percent[:100])
# plt.show()

num_components = 50
eigenVectors_subset = eigenVectors[:,0:num_components]

data_reduced = np.dot(eigenVectors_subset.transpose(),data_cent).transpose()
data = []
for i in range(len(data_reduced)):
    data.append([data_reduced[i,:],labels[i]])
# data_reduced = np.column_stack(data_reduced,labels)




print("Done!")
