import numpy as np


class KNN():
    """
    Each row of data is a sample and each column is a feature
    Last column of the data is the labels

    return: Matrix comparing expected results and predicted results, accuracy
    """

    def __init__(self, num_neighbours):
        self.num_neighbours = num_neighbours
        self.distances = []
        self.neighbours = []
        self.predictions = []

    def euclidean_distance(self, row1, row2):
        return np.linalg.norm(row1 - row2)

    def get_neighbors(self, train_dat, test_row):

        for train_row in train_dat:
            dist = self.euclidean_distance(test_row, train_row[:-1])
            self.distances.append((train_row, dist))

        self.distances.sort(key=lambda tup: tup[1])

        self.neighbours = []
        for i in range(self.num_neighbours):
            self.neighbours.append(self.distances[i][0])

    def predict(self, train_data, test_data):

        for i in range(len(test_data)):

            test_row = test_data[i,:-1]
            self.get_neighbors(train_data, test_row)

            output_values = [row[-1] for row in self.neighbours]
            prediction = max(set(output_values), key=output_values.count)
            self.predictions.append(prediction)

        return self.predictions
