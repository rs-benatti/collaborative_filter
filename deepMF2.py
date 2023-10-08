import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math


# Load the input data from a numpy file
train_data = np.load('dataset/ratings_train.npy')
# Replace NaN values with 0
train_data[np.isnan(train_data)] = 0
normalized_input_data = train_data/np.max(train_data)

test_data = np.load('dataset/ratings_test.npy')
test_data[np.isnan(test_data)] = 0
normalized_test_data = test_data/np.max(test_data)

encoded_dim = 10

# Define the model
class ParallelLayersModel(nn.Module):
    def __init__(self, input_size, hidden_size_row, hidden_size_col):
        super(ParallelLayersModel, self).__init__()
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
        #print(rows_output.shape) 
        cols_output = torch.relu(self.col_layer(cols))
        cols_output = torch.relu(self.col_layer2(cols_output))
        cols_output = torch.relu(self.col_layer3(cols_output))
        cols_output = torch.relu(self.col_output_layer(cols_output))
        #print(cols_output.shape)
        Y_hat = torch.mm(rows_output, cols_output.T)

        cols_output = torch.clamp(cols_output, min=0.0000001)
        rows_output = torch.clamp(rows_output, min=0.0000001)

        row_norms = torch.norm(rows_output, dim=1)
        cols_norms = torch.norm(cols_output, dim=1)
        # Compute the matrix of products using broadcasting
        #product_matrix = row_norms[:, None] * cols_norms
        product_matrix = torch.mm(row_norms[:, None], cols_norms[None, :])

        #print(f"min product {torch.min(product_matrix)}")
        Y_hat = Y_hat/product_matrix
        Y_hat = torch.clamp(Y_hat, max=0.99999, min=0.00001)
        return Y_hat

# Create an instance of the model
input_size = train_data.shape 
hidden_size_row = 16
hidden_size_col = 64
model = ParallelLayersModel(input_size, hidden_size_row, hidden_size_col)

# Define the training function
def train_model(model, input_data, num_epochs=100, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    weight = input_data.clone()
    weight[weight!=0] = 1
    weight[weight==0] = 0.01
    loss_fn = nn.BCELoss(weight=weight, reduction='mean')
    #loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
       
        Y_hat = model(input_data, input_data.T)
        #print(similarity_scores.shape)
        #print(torch.max(similarity_scores))
        #print(torch.min(similarity_scores))
        #print(torch.max(labels))
        #print(torch.min(labels))
        loss = loss_fn(Y_hat, input_data)
        loss.backward()
        optimizer.step()
        print(Y_hat)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')
            
    print('Training complete.')


# Training the model
train_model(model, torch.FloatTensor(normalized_input_data))
# Pass the new input data through the trained model to get predictions

predicted_similarity_scores = model(torch.FloatTensor(normalized_input_data), torch.FloatTensor(normalized_input_data).T)
print(torch.max(predicted_similarity_scores))
# Print or use the predicted_similarity_scores as needed
target = torch.FloatTensor(test_data)
print(torch.mean(torch.sqrt((predicted_similarity_scores[target!=0]*5 - target[target!=0])**2)))

target = torch.FloatTensor(train_data)
print(torch.mean(torch.sqrt((predicted_similarity_scores[target!=0]*5 - target[target!=0])**2)))