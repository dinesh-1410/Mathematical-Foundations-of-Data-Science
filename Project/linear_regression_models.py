#!/usr/bin/env python3
"""
Linear Regression Models Implementation
=====================================

This module implements three robust linear regression methods:
1. Ordinary Least Squares (OLS)
2. Least Median Squares (LMS)
3. Least Trimmed Squares (LTS)

Author: BE21B032
Course: Mathematical Foundations of Data Science (MFDS)
Institution: IIT Madras
Faculty: Dr. Arun Tangirala
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class LinearRegression:
    """
    A comprehensive linear regression class implementing OLS, LMS, and LTS methods.
    
    This class provides robust estimation methods that are resistant to outliers
    and can handle various types of data distributions.
    """
    
    def __init__(self, method='OLS', learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Initialize the Linear Regression model.
        
        Parameters:
        -----------
        method : str
            Estimation method: 'OLS', 'LMS', or 'LTS'
        learning_rate : float
            Learning rate for gradient descent
        max_iterations : int
            Maximum number of iterations for convergence
        tolerance : float
            Convergence tolerance
        """
        self.method = method.upper()
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Model parameters
        self.beta = None
        self.beta_history = []
        self.cost_history = []
        self.is_fitted = False
        
        # Validation
        if self.method not in ['OLS', 'LMS', 'LTS']:
            raise ValueError("Method must be 'OLS', 'LMS', or 'LTS'")
    
    def compute_cost(self, X, y, beta):
        """
        Compute the cost function for the given method.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Target vector (n_samples,)
        beta : np.ndarray
            Parameter vector (n_features,)
            
        Returns:
        --------
        float : Cost value
        """
        n_samples = len(y)
        predictions = X.dot(beta)
        residuals = (y - predictions) ** 2
        
        if self.method == 'OLS':
            # Ordinary Least Squares: minimize sum of squared residuals
            cost = np.sum(residuals) / (2 * n_samples)
            
        elif self.method == 'LMS':
            # Least Median Squares: minimize median of squared residuals
            cost = np.median(residuals)
            
        elif self.method == 'LTS':
            # Least Trimmed Squares: minimize sum of smallest q residuals
            q = int((n_samples / 2) + 1)
            sorted_residuals = np.sort(residuals)
            cost = np.sum(sorted_residuals[:q])
            
        return cost
    
    def compute_gradient(self, X, y, beta):
        """
        Compute the gradient of the cost function.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        beta : np.ndarray
            Parameter vector
            
        Returns:
        --------
        np.ndarray : Gradient vector
        """
        n_samples, n_features = X.shape
        predictions = X.dot(beta)
        residuals = y - predictions
        
        if self.method == 'OLS':
            # Standard OLS gradient
            gradient = (-2 / n_samples) * X.T.dot(residuals)
            
        elif self.method == 'LMS':
            # LMS gradient using sub-gradient approach
            squared_residuals = residuals ** 2
            median_residual = np.median(squared_residuals)
            
            # Find samples closest to median
            median_indices = np.where(np.abs(squared_residuals - median_residual) < 1e-10)[0]
            if len(median_indices) == 0:
                median_indices = [np.argmin(np.abs(squared_residuals - median_residual))]
            
            # Compute gradient for median samples
            gradient = np.zeros(n_features)
            for idx in median_indices:
                gradient += (-2) * residuals[idx] * X[idx, :]
            gradient /= len(median_indices)
            
        elif self.method == 'LTS':
            # LTS gradient using trimmed residuals
            squared_residuals = residuals ** 2
            q = int((n_samples / 2) + 1)
            threshold = np.partition(squared_residuals, q-1)[q-1]
            
            # Only consider residuals below threshold
            mask = squared_residuals <= threshold
            gradient = (-2) * X[mask].T.dot(residuals[mask])
            
        return gradient
    
    def fit(self, X, y, verbose=False):
        """
        Fit the linear regression model using gradient descent.
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Feature matrix
        y : np.ndarray or pd.Series
            Target vector
        verbose : bool
            Whether to print progress information
            
        Returns:
        --------
        self : LinearRegression
            Fitted model instance
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Add bias term (intercept)
        X = np.column_stack([np.ones(X.shape[0]), X])
        
        # Initialize parameters
        n_features = X.shape[1]
        self.beta = np.random.randn(n_features) * 0.01
        
        # Gradient descent
        for iteration in range(self.max_iterations):
            # Compute cost and gradient
            cost = self.compute_cost(X, y, self.beta)
            gradient = self.compute_gradient(X, y, self.beta)
            
            # Store history
            self.cost_history.append(cost)
            self.beta_history.append(self.beta.copy())
            
            # Update parameters
            self.beta -= self.learning_rate * gradient
            
            # Check convergence
            if iteration > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
                
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.6f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Feature matrix
            
        Returns:
        --------
        np.ndarray : Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Add bias term
        X = np.column_stack([np.ones(X.shape[0]), X])
        
        return X.dot(self.beta)
    
    def get_coefficients(self):
        """Get the fitted coefficients."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting coefficients")
        return self.beta
    
    def plot_convergence(self):
        """Plot the cost function convergence."""
        if not self.cost_history:
            raise ValueError("No training history available")
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title(f'{self.method} Cost Function Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True, alpha=0.3)
        plt.show()


def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance using multiple metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
        
    Returns:
    --------
    dict : Dictionary containing evaluation metrics
    """
    # Mean Squared Error
    mse = mean_squared_error(y_true, y_pred)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    
    # Relative Bias
    rb = np.mean((y_pred - y_true) / y_true) * 100
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Relative_Bias_%': rb,
        'R2': r2
    }


def load_insurance_data():
    """
    Load and preprocess the medical insurance dataset.
    
    Returns:
    --------
    tuple : (X, y) feature matrix and target vector
    """
    try:
        # Load data
        data = pd.read_csv('medical_insurance.csv')
        
        # Handle categorical variables
        categorical_cols = ['sex', 'smoker', 'region']
        data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        
        # Separate features and target
        X = data_encoded.drop('charges', axis=1)
        y = data_encoded['charges']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y.values, scaler
        
    except FileNotFoundError:
        print("medical_insurance.csv not found. Using synthetic data instead.")
        # Generate synthetic data for demonstration
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        
        X = np.random.randn(n_samples, n_features)
        true_beta = np.array([2.5, -1.2, 0.8, 1.5, -0.5])
        y = X.dot(true_beta) + np.random.normal(0, 0.5, n_samples)
        
        return X, y, None


def main():
    """
    Main function demonstrating the linear regression models.
    """
    print("=" * 60)
    print("Linear Regression Models Implementation")
    print("Mathematical Foundations of Data Science (MFDS)")
    print("IIT Madras - Dr. Arun Tangirala")
    print("=" * 60)
    
    # Load data
    print("\nLoading insurance dataset...")
    X, y, scaler = load_insurance_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    
    # Initialize models
    methods = ['OLS', 'LMS', 'LTS']
    models = {}
    results = {}
    
    print("\n" + "=" * 60)
    print("Training Models")
    print("=" * 60)
    
    for method in methods:
        print(f"\nTraining {method} model...")
        
        # Create and train model
        model = LinearRegression(method=method, learning_rate=0.01, max_iterations=1000)
        model.fit(X_train, y_train, verbose=True)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model
        metrics = evaluate_model(y_test, y_pred)
        
        # Store results
        models[method] = model
        results[method] = metrics
        
        print(f"{method} Training completed!")
        print(f"Final cost: {model.cost_history[-1]:.6f}")
    
    # Display results
    print("\n" + "=" * 60)
    print("Model Performance Comparison")
    print("=" * 60)
    
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))
    
    # Plot convergence for each method
    print("\nPlotting convergence curves...")
    plt.figure(figsize=(15, 5))
    
    for i, method in enumerate(methods):
        plt.subplot(1, 3, i+1)
        plt.plot(models[method].cost_history)
        plt.title(f'{method} Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Feature importance analysis (for OLS)
    if 'OLS' in models:
        print("\n" + "=" * 60)
        print("Feature Importance Analysis (OLS)")
        print("=" * 60)
        
        ols_model = models['OLS']
        coefficients = ols_model.get_coefficients()
        
        # Create feature names
        if scaler:
            feature_names = ['Intercept'] + [f'Feature_{i+1}' for i in range(len(coefficients)-1)]
        else:
            feature_names = ['Intercept'] + [f'Feature_{i+1}' for i in range(len(coefficients)-1)]
        
        # Display coefficients
        for name, coef in zip(feature_names, coefficients):
            print(f"{name:15}: {coef:8.4f}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    
    return models, results


if __name__ == "__main__":
    models, results = main()
