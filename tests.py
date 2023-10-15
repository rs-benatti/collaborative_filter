import numpy as np
import torch
import deepMF as dmf
import data_completer as dc
import matplotlib.pyplot as plt

#dc.complete_data()

# Load the input data from a numpy file
ratings_train = np.load("./dataset/ratings_train_completed.npy")
ratings_train2 = np.load('dataset/ratings_train.npy')
ratings_test = np.load('dataset/ratings_test.npy')
added_values_mask = np.load("./dataset/completed_mask.npy")

# Replace NaN values with 0
ratings_train[np.isnan(ratings_train)] = 0
ratings_train2[np.isnan(ratings_train2)] = 0
ratings_test[np.isnan(ratings_test)] = 0

weights = np.zeros(ratings_test.shape)
weights[added_values_mask is True] = 0.01
weights[ratings_train != 0] = 1

weights2 = np.zeros(ratings_train2.shape)
weights2[ratings_train2 != 0] = 1

"""total_data = ratings_train + ratings_test

non_empty_indices = np.where(total_data != 0)
non_empty_indices = np.array([non_empty_indices[0], non_empty_indices[1]])

indices = np.random.choice(np.array(range(non_empty_indices.shape[1])), size=non_empty_indices.shape[1], replace = False)

train_set_indices = indices[0:int(non_empty_indices.shape[1]*0.8)]
train_set_indices2d = np.array(non_empty_indices[:, train_set_indices])
train_set = np.zeros(total_data.shape)
train_set[train_set_indices2d[0], train_set_indices2d[1]] = total_data[train_set_indices2d[0], train_set_indices2d[1]]

test_set_indices = indices[int(non_empty_indices.shape[1]*0.8):]
test_set_indices2d = np.array(non_empty_indices[:, test_set_indices])
test_set = np.zeros(total_data.shape)
test_set[test_set_indices2d[0], test_set_indices2d[1]] = total_data[test_set_indices2d[0], test_set_indices2d[1]]"""

normalized_train_data = ratings_train / np.max(ratings_train)
normalized_train_data2 = ratings_train / np.max(ratings_train2)

normalized_test_data = ratings_test / np.max(ratings_test)

# Create an instance of the model
input_size = ratings_train.shape
hidden_size_row = 16
hidden_size_col = 64

# Training the model
optimizers = []
"""model = ParallelLayersModel(input_size, hidden_size_row, hidden_size_col)
optimizers.append((model, optim.Adam(model.parameters(), lr=0.001)))
model = ParallelLayersModel(input_size, hidden_size_row, hidden_size_col)
optimizers.append((model, optim.RAdam(model.parameters(), lr=0.001)))
model = ParallelLayersModel(input_size, hidden_size_row, hidden_size_col)
optimizers.append((model, optim.AdamW(model.parameters(), lr=0.001)))
model = ParallelLayersModel(input_size, hidden_size_row, hidden_size_col)
optimizers.append((model, optim.Rprop(model.parameters(), lr=0.001)))"""
for i in range(10):
    model = dmf.ParallelLayersModel(input_size, hidden_size_row, hidden_size_col)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    model2 = dmf.ParallelLayersModel(input_size, hidden_size_row, hidden_size_col)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.001, weight_decay=0.0001)

    dmf.train_model(model=model, optimizer=optimizer, input_data=torch.FloatTensor(normalized_train_data),
                    weights=torch.FloatTensor(weights), num_epochs=130, test_data=ratings_test)
    dmf.train_model(model=model2, optimizer=optimizer2, input_data=torch.FloatTensor(normalized_train_data2),
                    weights=torch.FloatTensor(weights2), num_epochs=130, test_data=ratings_test)
    # Pass the new input data through the trained model to get predictions
    predicted = model(torch.FloatTensor(normalized_train_data),
                      torch.FloatTensor(normalized_train_data).T)
    predicted2 = model(torch.FloatTensor(normalized_train_data2),
                       torch.FloatTensor(normalized_train_data2).T)

    plt.plot(model.rmse_test_hist, color="r")
    plt.plot(model2.rmse_test_hist, color="b")

plt.legend(loc="upper right")
plt.show()
