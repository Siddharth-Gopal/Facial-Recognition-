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

# #first index is the subject number and second index is pose number
# for j in test_set:
#     sub_pose = []
#     for i in range(0,13):
#         sub_pose.append([data[:,:,i,j].flatten(),i])
#     test_sub.append(sub_pose)

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


train_dat, test_dat = create_data()
k_col = []
acc_col = []

for k in range(1,51):
    count = 0
    for i in range(0, np.shape(test_dat)[0]):
        prediction = predict_classification(train_dat, test_dat[i], k)
        print('Expected %d, Got %d.' % (test_dat[i][1], prediction))
        if test_dat[i][1] == prediction : count = count + 1
    print(f'Accuracy is {count/np.shape(test_dat)[0]}')
    k_col.append(k)
    acc_col.append(count/np.shape(test_dat)[0])

plt.scatter(k_col,acc_col)
plt.show()
# neighbours = get_neighbors(train_dat, test_dat[0], 3)
# plt.imshow(test_dat[0][0].reshape(48,40))
# plt.show()
# plt.imshow(neighbours[0][0].reshape(48,40))
# plt.show()
# plt.imshow(neighbours[1][0].reshape(48,40))
# plt.show()
# plt.imshow(neighbours[2][0].reshape(48,40))
# plt.show()


print("Done!")
