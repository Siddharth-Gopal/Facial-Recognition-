import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.io import loadmat
import random

random.seed(5)
train_set = random.sample(range(0, 400), 350)
test_set = list(set(train_set) ^ set(range(0, 400)))

annots = loadmat('data.mat')
face_data = annots["face"]
neutral_face = []
expression_face = []
for i in range(0, int(np.shape(face_data)[2] / 3)):
    neutral_face.append([face_data[:, :, 3 * i].flatten(), 1])
    expression_face.append([face_data[:, :, (3 * i) + 1].flatten(), -1])

data = np.vstack((neutral_face,expression_face))

X = np.empty(len(data[0][0]))
Y = []
for i in range(len(data)):
    X = np.vstack((X,data[i][0]))
    Y.append(data[i][1])

X = X[1:,:]
Y = np.array(Y)

def euclidean_distance(row1, row2):
    row1 = row1[0]
    row2 = row2[0]
    return np.linalg.norm(row1 - row2)


def create_data():
    test_dat = []
    for i in test_set:
        test_dat.append(data[i])

    random.shuffle(test_dat)

    train_dat = []
    for i in train_set:
        train_dat.append(data[i])

    random.shuffle(train_dat)

    return train_dat, test_dat


def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


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


train_dat, test_dat = create_data()
k_col = []
acc_col = []

# for different number of neighbours
for k in range(1, 101):
    count = 0

    # for different samples of test data
    for i in range(0, np.shape(test_dat)[0]):
        prediction = predict_classification(train_dat, test_dat[i], k)
        print('Expected %d, Got %d.' % (test_dat[i][1], prediction))
        if test_dat[i][1] == prediction: count = count + 1

    print(f'Accuracy is {count / np.shape(test_dat)[0]}')
    k_col.append(k)
    acc_col.append(count / np.shape(test_dat)[0])

plt.scatter(k_col, acc_col)
plt.show()

print("Done!")
