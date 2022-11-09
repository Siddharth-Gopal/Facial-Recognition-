import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.io import loadmat
import random

random.seed(5)
train_set = random.sample(range(0,68),50)
test_set = list(set(train_set) ^ set(range(0,68)))
test_sub = []

annots = loadmat('pose.mat')
data = annots["pose"]

def euclidean_distance(row1, row2):
    row1 = row1[0]
    row2 = row2[0]
    return np.linalg.norm(row1 - row2)

def create_data():
    test_dat = []
    for j in test_set:
        for i in range(0,13):
            test_dat.append([data[:,:,i,j].flatten(),i])

    random.shuffle(test_dat)

    train_dat = []
    for j in train_set:
        for i in range(0,13):
            train_dat.append([data[:,:,i,j].flatten(),i])

    random.shuffle(train_dat)

    return train_dat, test_dat

def get_neighbors(train_dat, test_row, num_neighbors):
    distances = list()
    for train_row in train_dat:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

def remove_label(train_dat):
    data = []
    labels = []
    for i in range(np.shape(train_dat)[0]):
        data.append(train_dat[i][0])
        labels.append(train_dat[i][1])
    return np.array(data),np.array(labels)

def center_values(data):
    data_cent = np.zeros(np.shape(data)[0])
    center_function = lambda x: x - x.mean()
    for i in range(np.shape(data)[1]):
        col = center_function(data[:,i])
        data_cent = np.column_stack((data_cent,col))
        # print("Added Column")
    return data_cent[:,1:]

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

def test_data_centre(test_dat, mean):
    data = []
    for i in range(np.shape(test_dat)[0]):
        row = test_dat[i,:]
        data.append(row - mean)

    data = np.array(data)
    return data

def data_grouping(data,labels):
    """
    Groups the data according to its labels
    :param data: The complete data matrix with labels
    :param labels: The labels for which the data is to be grouped. Can also input the labels list created with create_data_set()
    :return: a list of the grouped data
    """

    data_groups = []
    def group_data(data, label):
        data_grouped=np.empty(len(data[0][0]))
        for i in range(len(data)):
            if data[i][1]==label:
                data_grouped = np.vstack((data_grouped,data[i][0]))
        data_grouped = data_grouped[1:,:]
        # data_grouped = np.column_stack((data_grouped,np.full(len(data_grouped),label)))
        return data_grouped

    for i in set(labels):
        data_groups.append(group_data(data, label=i))

    return data_groups

train_dat, test_dat = create_data()
data, labels = remove_label(train_dat)
test_dat, test_labels = remove_label(test_dat)

mean = data_mean(data)

data_cent = center_values(data)

data_cent_w_labels = []
for i,row in enumerate(data_cent):
    data_cent_w_labels.append([row,labels[i]])

data_group = data_grouping(data_cent_w_labels, labels)

SW = np.zeros((np.shape(data)[1], np.shape(data)[1]))
SB = np.zeros((np.shape(data)[1], np.shape(data)[1]))

for group in data_group:
    group_mean = (data_mean(group))
    diff = group-group_mean
    SW += diff.T.dot(diff)

    num_classes = len(data_group)
    mean_diff = (group_mean - mean).reshape(np.shape(data)[1],1)
    SB += num_classes * (mean_diff).dot(mean_diff.T)


A = np.linalg.inv(SW).dot(SB)

eigenvalues, eigenvectors = np.linalg.eigh(A)
eigenvectors = eigenvectors.T

idxs = np.argsort(abs(eigenvalues))[::-1]
eigenvalues = eigenvalues[idxs]
eigenvectors = eigenvectors[idxs]

num_components = 300

linear_discriminants = eigenvectors[0: num_components]

data_reduced = np.dot(data_cent, linear_discriminants.T)

test_data = test_data_centre(test_dat,mean)

test_dat = np.dot(test_data, linear_discriminants.T)

train_data = []
for i in range(len(data_reduced)):
    train_data.append([data_reduced[i,:],labels[i]])

test_data = []
for i in range(len(test_dat)):
    test_data.append([test_dat[i,:],test_labels[i]])

k_col = []
acc_col = []

for k in range(1,31):
    count = 0
    for i in range(0, np.shape(test_data)[0]):
        prediction = predict_classification(train_data, test_data[i], k)
        print('Expected %d, Got %d.' % (test_data[i][1], prediction))
        if test_data[i][1] == prediction : count = count + 1
    print(f'Accuracy is {count/np.shape(test_data)[0]}')
    k_col.append(k)
    acc_col.append(count/np.shape(test_data)[0])

plt.scatter(k_col,acc_col)
plt.show()

print("Done")