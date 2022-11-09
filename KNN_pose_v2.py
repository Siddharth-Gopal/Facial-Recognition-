import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import random
from algorithms.KNN import KNN
from Utils.misc import *

random.seed(5)
train_set = random.sample(range(0,68),50)
test_set = list(set(train_set) ^ set(range(0,68)))
test_sub = []

annots = loadmat('pose.mat')
data = annots["pose"]


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

train_data, test_data = create_pose_data_set()

knn = KNN(9)
predictions = knn.predict(train_data, test_data)

test_data_wout_labels, labels = remove_label(test_data)

accuracy = accuracy_score(labels,predictions)

print(accuracy)