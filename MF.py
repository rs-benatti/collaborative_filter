import numpy as np
from scipy.stats import norm, fit
from scipy.optimize import fsolve



class MatrixFactorizarion:
    def __init__(self, R, l, mu, k):
        self.R = R
        self.k = k
        self.l = l
        self.mu = mu
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
        
        self.I = np.random.normal(mu_UI, sigma, (self.R.shape[0], self.k)).astype('float64')
        self.U = np.random.normal(mu_UI, sigma, (self.R.shape[1], self.k)).astype('float64')

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
    
    def fit(self, lr_I, lr_U, num_iterations):
        # Compute ∂C/∂U(I, U)
        # -2RᵀI + 2UIᵀI + 2µU
        R = self.R
        R = np.nan_to_num(R, nan=0)

        mu = self.mu
        l = self.l
        
        for iteration in range(num_iterations):
            U = self.U
            I = self.I
            U_grad = -2 * np.dot(R.T, I) + 2 * np.dot(U, np.dot(I.T, I)) + 2 * mu * U
            I_grad = -2 * np.dot(R, U) + 2 * np.dot(I, np.dot(U.T, U)) + 2 * l * I
            self.I -= lr_I*I_grad
            self.U -= lr_U*U_grad
            
            cost = self.C(R, I, U, l, mu)

            print(f"Iteration {iteration + 1}: Cost = {cost}")

    def predict(self):
        return self.I @ self.U.T

