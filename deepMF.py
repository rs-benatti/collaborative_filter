import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from time import time
import data_completer as dc

# Define the model
class ParallelLayersModel(nn.Module):
    def __init__(self, input_size, hidden_size_row, hidden_size_col, encoded_dim=10):
        super(ParallelLayersModel, self).__init__()
        self.rmse_train_hist = []
        self.rmse_test_hist = []
        self.row_layer = nn.Linear(input_size[1], hidden_size_row*2)
        self.row_layer2 = nn.Linear(hidden_size_row*2, hidden_size_row)
        self.row_layer3 = nn.Linear(hidden_size_row, int(hidden_size_row/2))

        self.col_layer = nn.Linear(input_size[0], hidden_size_col*2)
        self.col_layer2 = nn.Linear(hidden_size_col*2, hidden_size_col)
        self.col_layer3 = nn.Linear(hidden_size_col, int(hidden_size_col/2))

        self.row_output_layer = nn.Linear(int(hidden_size_row/2), encoded_dim)
        self.col_output_layer = nn.Linear(int(hidden_size_col/2), encoded_dim)

        
    def forward(self, rows, cols):
        rows_output = torch.relu(self.row_layer(rows))
        rows_output = torch.relu(self.row_layer2(rows_output))
        rows_output = torch.relu(self.row_layer3(rows_output))
        rows_output = torch.relu(self.row_output_layer(rows_output))

        cols_output = torch.relu(self.col_layer(cols))
        cols_output = torch.relu(self.col_layer2(cols_output))
        cols_output = torch.relu(self.col_layer3(cols_output))
        cols_output = torch.relu(self.col_output_layer(cols_output))
    
        Y_hat = torch.mm(rows_output, cols_output.T)

        cols_output = torch.clamp(cols_output, min=0.0000001)
        rows_output = torch.clamp(rows_output, min=0.0000001)

        row_norms = torch.norm(rows_output, dim=1)
        cols_norms = torch.norm(cols_output, dim=1)
        # Compute the matrix of products using broadcasting
        product_matrix = torch.mm(row_norms[:, None], cols_norms[None, :])

        Y_hat = Y_hat/product_matrix
        Y_hat = torch.clamp(Y_hat, max=0.99999, min=0.00001)
        return Y_hat
    
    def RMSE(self, Y, Y_hat):
        return torch.mean(torch.sqrt((Y_hat[Y!=0] - Y[Y!=0])**2)).item()
    
    def numpy_and_round(self, Y_hat):# Y_hat normalized
        return np.round(Y_hat.detach().numpy() * 10)/2
    
    def to_numpy(self, Y_hat):
        return Y_hat.detach() * 5
        

# Define the training function
def train_model(model, optimizer, input_data, weights, num_epochs=250, test_data=False): # Obs.: test_data must not be normalized   
    if (test_data is not False):
        target_train = torch.FloatTensor(input_data*5)
        target_test = torch.FloatTensor(test_data)
    loss_fn = nn.BCELoss(weight=weights, reduction='mean')
    #loss_fn = nn.MSELoss()
    rmse_train = []
    rmse_test = []
    times = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()      
        Y_hat = model(input_data, input_data.T)
        loss = loss_fn(Y_hat, input_data)
        loss.backward()
        optimizer.step()

        if test_data is not False:
            rmse_train.append(model.RMSE(target_train, Y_hat*5))     
                   
            rmse_test.append(model.RMSE(target_test, Y_hat*5))
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

        
    model.rmse_train_hist = rmse_train
    model.rmse_test_hist = rmse_test

    plt.plot(rmse_test)
    
    print('Training complete.')



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
model = ParallelLayersModel(input_size, hidden_size_row, hidden_size_col)
optimizers.append((model, optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)))




for i in range(len(optimizers)):
    train_model(optimizers[i][0], optimizers[i][1], torch.FloatTensor(normalized_train_data), weights=torch.FloatTensor(weights))
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



