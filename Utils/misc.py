import numpy as np

def llnorm(sample, mu,sigma):
    sample_centre = sample - mu

    ll = -0.5 * np.linalg.det(sigma) - 0.5*np.linalg.multi_dot([sample_centre.transpose(), np.linalg.inv(sigma), sample_centre])
    return ll

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

def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy