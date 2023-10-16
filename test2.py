import numpy as np
import deepMF
import torch

# Load the input data from a numpy file
ratings_train = np.load("./dataset/ratings_train.npy")
ratings_test = np.load("dataset/ratings_test.npy")

# Replace NaN values with 0
ratings_train[np.isnan(ratings_train)] = 0
ratings_test[np.isnan(ratings_test)] = 0
total_data = ratings_train + ratings_test

non_empty_indices = np.where(total_data != 0)
non_empty_indices = np.array([non_empty_indices[0], non_empty_indices[1]])

indices = np.random.choice(np.array(range(non_empty_indices.shape[1])), size=non_empty_indices.shape[1],
                           replace=False)

train_set_indices = indices[0:int(non_empty_indices.shape[1] * 0.8)]
train_set_indices2d = np.array(non_empty_indices[:, train_set_indices])
train_set = np.zeros(total_data.shape)
train_set[train_set_indices2d[0], train_set_indices2d[1]] = total_data[
    train_set_indices2d[0], train_set_indices2d[1]]

test_set_indices = indices[int(non_empty_indices.shape[1] * 0.8):]
test_set_indices2d = np.array(non_empty_indices[:, test_set_indices])
test_set = np.zeros(total_data.shape)
test_set[test_set_indices2d[0], test_set_indices2d[1]] = total_data[test_set_indices2d[0], test_set_indices2d[1]]

normalised_train_set = train_set/np.max(train_set)

input_size = train_set.shape
encoded_dim = 32
hidden_size_row = 16
hidden_size_col = 64
model = deepMF.ParallelLayersModel(input_size, hidden_size_row, hidden_size_col, encoded_dim)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
deepMF.train_model(model, optimizer, torch.FloatTensor(normalised_train_set), False, num_epochs=130)
predicted = model(torch.FloatTensor(normalised_train_set), torch.FloatTensor(normalised_train_set).T)
target = torch.FloatTensor(test_set)
print(torch.sqrt(torch.mean((predicted[target!=0]*5 - target[target!=0])**2)), model.RMSE(torch.FloatTensor(test_set), predicted*5))