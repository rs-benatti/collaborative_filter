import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Load the input data from a numpy file
input_data = np.load('dataset/ratings_test.npy')
indices_Y_plus = np.where(~np.isnan(input_data))
Y_plus = np.array(np.where(~np.isnan(input_data))).T

# Replace NaN values with 0
input_data[np.isnan(input_data)] = 0

Y_plus_i = input_data[indices_Y_plus[0]] # each column that has a valid value
Y_plus_j = input_data[:, indices_Y_plus[1]] # each row that has a valid value
print(Y_plus_i.shape)
print(Y_plus_j.shape)
'''
neg_ratio = 5
total_neg_values = neg_ratio * Y_plus.shape[0]
Y_neg = [[np.random.randint(0, input_data.shape[0]), np.random.randint(0, input_data.shape[1])] for _ in range(total_neg_values)]
Y_neg = np.array(Y_neg)
Y_neg_i = input_data[Y_neg.T[0]]
Y_neg_j = input_data[:, Y_neg.T[1]]
#print(Y_neg_i.shape)
#print(Y_neg_j.shape)
'''
input_data = input_data/np.max(input_data)

normalized_input_data = input_data/np.max(input_data)
print(normalized_input_data.shape)



encoded_dim = 5

# Define the model
class ParallelLayersModel(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size):
        super(ParallelLayersModel, self).__init__()
        

        self.row_layer = nn.Linear(input_size1, hidden_size)
        self.col_layer = nn.Linear(input_size2, hidden_size)
        self.row_output_layer = nn.Linear(hidden_size, encoded_dim)
        self.col_output_layer = nn.Linear(hidden_size, encoded_dim)
        
    def forward(self, rows, cols):
        rows_output = torch.relu(self.row_layer(rows))
        rows_output = torch.relu(self.row_output_layer(rows_output))
        #print(rows_output.shape) 
        cols_output = torch.relu(self.col_layer(cols))
        cols_output = torch.relu(self.col_output_layer(cols_output))
        #print(cols_output.shape)

        similarity = torch.mm(rows_output, cols_output.T)
        row_norms = torch.norm(rows_output, dim=1)
        cols_norms = torch.norm(cols_output, dim=1)
        #print(row_norms.shape)
        #print(cols_norms.shape)

        # Compute the matrix of products using broadcasting
        product_matrix = row_norms[:, None] * cols_norms
        #print(f"min product {torch.min(product_matrix)}")
        similarity = similarity/product_matrix
        # Create a mask for NaN values
        nan_mask = torch.isnan(similarity)

        # Replace NaN values with 0
        similarity[nan_mask] = 0
        torch.sigmoid(similarity)
        return similarity

# Create an instance of the model
#input_size = input_data.shape 
input_size = (Y_plus_i.shape[1], Y_plus_j.shape[0])
Y_input = [Y_plus_i, Y_plus_j.T]
hidden_size = 32
model = ParallelLayersModel(input_size[0], input_size[1], hidden_size)

# Define the training function
def train_model(model, input_indices, labels, num_epochs=100, learning_rate=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        input_data1 = torch.FloatTensor(input_data[indices_Y_plus[0]])
        print(input_data[indices_Y_plus[0]])
        input_data2 = torch.FloatTensor(input_data[:, indices_Y_plus[1]].T)
        similarity_scores = model(input_data1, input_data2)
        #print(similarity_scores.shape)
        #print(torch.max(similarity_scores))
        #print(torch.min(similarity_scores))
        #print(torch.max(labels))
        #print(torch.min(labels))
        loss = criterion(similarity_scores, labels)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            
    print('Training complete.')

train_model(model, Y_plus, torch.FloatTensor(normalized_input_data))

'''
# Training the model
train_model(model, torch.FloatTensor(input_data), torch.FloatTensor(normalized_input_data))
# Pass the new input data through the trained model to get predictions
predicted_similarity_scores = model(torch.FloatTensor(input_data), torch.FloatTensor(normalized_input_data).T)
print(torch.max(predicted_similarity_scores))
# Print or use the predicted_similarity_scores as needed
print(predicted_similarity_scores)
'''