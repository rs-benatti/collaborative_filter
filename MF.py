import numpy as np
from scipy.stats import norm, fit
from scipy.optimize import fsolve



class MatrixFactorizarion:
    def __init__(self, R, l, mu, k):
        self.R = R
        self.k = k
        self.l = l
        self.mu = mu
        non_nan_indices = np.where(~np.isnan(self.R))
        self.non_nan_indices = non_nan_indices
        row_indices, col_indices = non_nan_indices
        mean = self.R[~np.isnan(self.R)].mean()
        self.S = [row_indices, col_indices, self.R[~np.isnan(self.R)]]
        values = self.R.flatten()
        values = values[~np.isnan(values)]
        # Fit a normal distribution to the histogram
        mu, std = norm.fit(values)
        mu_UI = mu**0.5
        # Define the equation
        def equation(sigma, var_R, mu_UI):
            return sigma**4 + 2 * mu_UI**2 * sigma**2 - var_R

        # Define the known values
        var_R = std**2

        initial_guess = 1.0
        sigma = fsolve(equation, initial_guess, args=(var_R, mu_UI))[0]
        
        self.I = (np.ones((self.R.shape[0], self.k)) / self.k**0.5) * mean**0.5
        self.U = (np.ones((self.R.shape[1], self.k)) / self.k**0.5) * mean**0.5
        print(self.I @ self.U.T)

    def C(self, R, I, U, l, mu):
        '''
        C =  # ∥R - IU^T∥² + λ∥I∥² + µ∥U∥²
        '''

        '''
        # ∥R - IU^T∥²

        loss_term = R - (I @ U.T)
        # Since nan - number returns nan, now we may replace our nans with 0 and calculate the Frobenius norm 
        loss_term = np.nan_to_num(loss_term, nan=0)
        
        # frobenius_norm(X)^2 = trace(X^T@X)

        loss_term = np.trace(loss_term.T@loss_term)

        #  λ∥I∥²
        I_penalization_term = l * np.trace(I.T @ I)

        #  µ∥U∥²
        U_penalization_term = mu * np.trace(U.T @ U)
        '''
        R = self.R
        R = np.nan_to_num(R, nan=0)
        #return loss_term + I_penalization_term + U_penalization_term
        return np.trace(R.T @ R) - 2 * np.trace(R.T @ I @ U.T) + np.trace(U @ I.T @ I @ U.T) + self.l * np.trace(I.T @ I) + self.mu * np.trace(U.T @ U)

    def calculate_gradient_iiq(self, i, q):
        gradient = 0
        non_nan_indices = np.column_stack(self.non_nan_indices)
        
        for non_nan_index in non_nan_indices[non_nan_indices[:, 0]==i]:
            j = non_nan_index[1]
            product = 0
            for s in range(self.k):
                product += self.I[i, s] * self.U[non_nan_index[1], s]
            gradient += (self.R[i, j] - product)*(-self.U[j, q]) + 2 * self.l * self.I[i, q]
        return 2*gradient

    def calculate_gradient_ujq(self, j, q):
        gradient = 0
        non_nan_indices = np.column_stack(self.non_nan_indices)
        
        for non_nan_index in non_nan_indices[non_nan_indices[:, 1]==j]:
            i = non_nan_index[0]
            product = 0
            for s in range(self.k):
                product += self.I[i, s] * self.U[j, s]
            gradient += (self.R[i, j] - product)*(-self.I[i, q]) + 2 * self.mu * self.U[j, q]
        return 2*gradient

    
    def fit(self, lr_I, lr_U, num_iterations):
        R = self.R
        R = np.nan_to_num(R, nan=0)

        mu = self.mu
        l = self.l
        '''
        for iteration in range(num_iterations):
            U = self.U
            I = self.I
            U_grad = -2 * np.dot(R.T, I) + 2 * np.dot(U, np.dot(I.T, I)) + 2 * mu * U
            I_grad = -2 * np.dot(R, U) + 2 * np.dot(I, np.dot(U.T, U)) + 2 * l * I
            self.I -= lr_I*I_grad
            self.U -= lr_U*U_grad
            
            cost = self.C(R, I, U, l, mu)

            print(f"Iteration {iteration + 1}: Cost = {cost}")
        '''
        for iteration in range(num_iterations):
            gradients_I = np.zeros_like(self.I)  # Initialize gradients for I matrix
            gradients_U = np.zeros_like(self.U)  # Initialize gradients for U matrix

            # Calculate gradients for I matrix
            num_users = self.R.shape[0]
            num_items = self.R.shape[1]
            '''
            for i in range(num_users):
                for q in range(self.k):
                    gradient_iiq = self.calculate_gradient_iiq(i, q)
                    gradients_I[i, q] = gradient_iiq
            # Calculate gradients for U matrix
            num_items = self.R.shape[1]
            for j in range(num_items):
                for q in range(self.k):
                    gradient_ujq = self.calculate_gradient_ujq(j, q)
                    gradients_U[j, q] = gradient_ujq
            '''
            for q in range(self.k):
                for i in range(num_users):
                    gradient_iiq = self.calculate_gradient_iiq(i, q)
                    gradients_I[i, q] = gradient_iiq
                for j in range(num_items):
                    gradient_ujq = self.calculate_gradient_ujq(j, q)
                    gradients_U[j, q] = gradient_ujq
            # Update I and U matrices using gradients and learning rate
            self.I -= lr_I * gradients_I
            self.U -= lr_U * gradients_U
            cost = self.C(self.R, self.I, self.U, l, mu)

            print(f"Iteration {iteration + 1}: Cost = {cost}")

    def predict(self):
        return self.I @ self.U.T

    def RMSE(self, predictions, targets):
        """
        Calculate the Root Mean Squared Error (RMSE) between two vectors.

        Parameters:
        - predictions: A NumPy array or list of predicted values.
        - targets: A NumPy array or list of target (true) values.

        Returns:
        - rmse_value: The RMSE between the two vectors.
        """
        # Ensure predictions and targets are NumPy arrays
        predictions = np.array(predictions)
        targets = np.array(targets)

        # Calculate the squared differences between predictions and targets
        squared_errors = (predictions - targets) ** 2

        # Calculate the mean of squared errors
        mean_squared_error = squared_errors.mean()

        # Calculate the square root to get RMSE
        rmse_value = np.sqrt(mean_squared_error)

        return rmse_value

