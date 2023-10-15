import numpy as np
import torch
import deepMF as dmf
import data_completer as dc
import matplotlib.pyplot as plt

# Load the input data from a numpy file
ratings_train = np.load("./dataset/ratings_train_completed.npy")
ratings_test = np.load('dataset/ratings_test.npy')
added_values_mask = np.load("./dataset/completed_mask.npy")
# Replace NaN values with 0
ratings_train[np.isnan(ratings_train)] = 0
ratings_test[np.isnan(ratings_test)] = 0

weights = np.zeros(ratings_test.shape)
weights[added_values_mask == True] = 0.1
weights[ratings_train != 0] = 1


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

normalized_train_data = ratings_train/np.max(ratings_train)

normalized_test_data = ratings_test/np.max(ratings_test)

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
model = dmf.ParallelLayersModel(input_size, hidden_size_row, hidden_size_col)
optimizers.append((model, torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)))




for i in range(len(optimizers)):
    dmf.train_model(optimizers[i][0], optimizers[i][1], torch.FloatTensor(normalized_train_data), weights=torch.FloatTensor(weights))
    # Pass the new input data through the trained model to get predictions
    predicted_similarity_scores = optimizers[i][0](torch.FloatTensor(normalized_train_data), torch.FloatTensor(normalized_train_data).T)
    target = torch.FloatTensor(ratings_train)
    print(torch.sqrt(torch.mean((predicted_similarity_scores[target!=0]*5 - target[target!=0])**2)))
    target = torch.FloatTensor(ratings_test)
    print(torch.sqrt(torch.mean((predicted_similarity_scores[target!=0]*5 - target[target!=0])**2)))
plt.legend(loc="upper right")
plt.show()


"""R = np.load('./dataset/ratings_train.npy')
A = np.load('./dataset/namesngenre.npy')
reworked_A = dc.rework_metadata(A)
years = dc.get_years(A)
distances_matrix = dc.distance_between_movies(reworked_A, years)
neighbors = dc.get_neighbors(distances_matrix)
max = 0
for ele in neighbors:
    if len(ele) > max:
        max = len(ele)
print(max)
mean = 0
for ele in neighbors:
    mean += len(ele)
print(mean/len(neighbors))
new_R, added_values = dc.add_ratings(R.copy(), neighbors)
np.save("./dataset/ratings_train_completed.npy", new_R)
np.save("./dataset/completed_mask.npy", added_values)
"""