import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt




# Define the model
class ParallelLayersModel(nn.Module):
    def __init__(self, input_size, hidden_size_row, hidden_size_col, encoded_dim=10):
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
input_size = train_set.shape
hidden_size_row = 16
hidden_size_col = 64

# Define the training function
def train_model(model, optimizer, input_data, weight_decay = False, num_epochs=400):    
    weight = input_data.clone()
    weight[weight!=0] = 1
    weight[weight==0] = 0.0000
    loss_fn = nn.BCELoss(weight=weight, reduction='mean')
    #loss_fn = nn.MSELoss()
    rmse_train = []
    rmse_test = []
    times = []
    t = time.time()
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

        target = torch.FloatTensor(train_set)
        rmse_train.append(torch.sqrt(torch.mean((Y_hat[target!=0]*5 - target[target!=0])**2)).item())

        target = torch.FloatTensor(test_set)
        rmse_test.append(torch.sqrt(torch.mean((Y_hat[target!=0]*5 - target[target!=0])**2)).item())
        times.append(time.time() - t)


        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')
    plt.plot(rmse_test, label=type (optimizer).__name__+" "+str(weight_decay)+" min: "
             +str(min(rmse_test))+" index: "+str(rmse_test.index(min(rmse_test))) +
               " time: "+str(times[rmse_test.index(min(rmse_test))]))
"""plt.plot(rmse_test, label=type (optimizer).__name__+" k="+str(k)+" min: "
             +str(min(rmse_test))+" index: "+str(rmse_test.index(min(rmse_test))) +
               " time: "+str(times[rmse_test.index(min(rmse_test))]))"""
    #plt.plot(rmse_test)
    
    #print('Training complete.')


# Training the model
optimizers = []
model = ParallelLayersModel(input_size, hidden_size_row, hidden_size_col)
optimizers.append((model, optim.Adam(model.parameters(), lr=0.001)))
model = ParallelLayersModel(input_size, hidden_size_row, hidden_size_col)
optimizers.append((model, optim.RAdam(model.parameters(), lr=0.001)))
model = ParallelLayersModel(input_size, hidden_size_row, hidden_size_col)
optimizers.append((model, optim.AdamW(model.parameters(), lr=0.001)))
model = ParallelLayersModel(input_size, hidden_size_row, hidden_size_col)
optimizers.append((model, optim.Rprop(model.parameters(), lr=0.001)))
model = ParallelLayersModel(input_size, hidden_size_row, hidden_size_col)
optimizers.append((model, optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)))

for i in range(len(optimizers)):
    train_model(optimizers[i][0], optimizers[i][1], torch.FloatTensor(normalized_train_data), weight_decay=i>=5)
    # Pass the new input data through the trained model to get predictions
    predicted_similarity_scores = optimizers[i][0](torch.FloatTensor(normalized_train_data), torch.FloatTensor(normalized_train_data).T)
    target = torch.FloatTensor(train_set)
    print(torch.sqrt(torch.mean((predicted_similarity_scores[target!=0]*5 - target[target!=0])**2)))
    target = torch.FloatTensor(test_set)
    print(torch.sqrt(torch.mean((predicted_similarity_scores[target!=0]*5 - target[target!=0])**2)))
plt.legend(loc="upper right")
plt.show()

"""for k in [25, 30, 35, 40]:
    model = ParallelLayersModel(input_size, hidden_size_row, hidden_size_col, encoded_dim=k)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, optimizer, torch.FloatTensor(normalized_train_data))
    # Pass the new input data through the trained model to get predictions
    predicted_similarity_scores = model(torch.FloatTensor(normalized_train_data), torch.FloatTensor(normalized_train_data).T)
    target = torch.FloatTensor(train_set)
    print(torch.mean(torch.sqrt((predicted_similarity_scores[target!=0]*5 - target[target!=0])**2)))
    target = torch.FloatTensor(test_set)
    print(torch.mean(torch.sqrt((predicted_similarity_scores[target!=0]*5 - target[target!=0])**2)))"""
"""rmseValues = []
for i in range(10):
    model = ParallelLayersModel(input_size, hidden_size_row, hidden_size_col, encoded_dim=32)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    train_model(model, optimizer, torch.FloatTensor(normalized_train_data), num_epochs=130)
    # Pass the new input data through the trained model to get predictions
    predicted_similarity_scores = model(torch.FloatTensor(normalized_train_data), torch.FloatTensor(normalized_train_data).T)
    target = torch.FloatTensor(train_set)
    print("train rmse", torch.mean(torch.sqrt((predicted_similarity_scores[target!=0]*5 - target[target!=0])**2)))
    target = torch.FloatTensor(test_set)
    rmseValues.append(torch.mean(torch.sqrt((predicted_similarity_scores[target!=0]*5 - target[target!=0])**2)).item())
    print("test rmse", rmseValues[-1])
    print(i+1)
avgrmse = sum(rmseValues)/len(rmseValues)
print("avg rmse", avgrmse)
plt.plot(np.ones(130) * avgrmse)
#plt.legend(loc="upper right")
plt.show()"""