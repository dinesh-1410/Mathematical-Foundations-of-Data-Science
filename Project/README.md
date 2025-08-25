# MFDS Course Project Implementation

This directory contains the main project implementations for the Mathematical Foundations of Data Science course.

##  Project Overview

The course project focuses on implementing and comparing robust statistical methods for real-world data analysis, specifically:

1. **Linear Regression Models** - Implementation of OLS, LMS, and LTS methods
2. **Lomb-Scargle Periodogram** - Time series periodicity detection
3. **Insurance Analytics** - Real-world application of robust regression

##  Files Description

### Core Implementations

#### `linear_regression_models.py`
- **Purpose**: Implementation of three robust linear regression methods
- **Methods**: OLS (Ordinary Least Squares), LMS (Least Median Squares), LTS (Least Trimmed Squares)
- **Features**:
  - Custom gradient descent implementation
  - Robust estimation for outlier-resistant analysis
  - Comprehensive evaluation metrics (MSE, RB, MAE, R²)
  - Insurance data analysis application

#### `lomb_scargle_periodogram.py`
- **Purpose**: Time series periodicity detection using Lomb-Scargle method
- **Application**: Tesla stock price data analysis
- **Features**:
  - Unevenly sampled time series support
  - Peak detection and analysis
  - Signal reconstruction and validation
  - Comparison with synthetic data

### Data Files

#### `medical_insurance.csv`
- **Source**: Medical insurance dataset
- **Features**: Age, sex, BMI, children, smoker status, region, charges
- **Target**: Insurance charges (regression target)
- **Usage**: Training and testing robust regression models

#### `Tesla Stock Price.csv`
- **Source**: Tesla stock price historical data
- **Features**: Date, price information
- **Usage**: Periodicity analysis and time series modeling

### Documentation

#### `BE21B032_CH5019_PROJECT.pdf`
- **Content**: Complete project report and methodology
- **Sections**: Problem formulation, mathematical foundations, implementation details, results analysis

## 🛠️ Implementation Details

### Linear Regression Models

#### Mathematical Foundation

**OLS (Ordinary Least Squares):**
```
β̂_OLS = argmin_β Σ(y_i - x_i^T β)²
```

**LMS (Least Median Squares):**
```
β̂_LMS = argmin_β Med(y_i - x_i^T β)²
```

**LTS (Least Trimmed Squares):**
```
β̂_LTS = argmin_β Σ_{i=1}^q r_{(i)}(β)²
```
where r_{(1)}(β)² ≤ ... ≤ r_{(n)}(β)² are ordered squared residuals and q = (n/2 + 1).

#### Algorithm Implementation

1. **Gradient Computation**: Custom gradients for each method
2. **Optimization**: Gradient descent with adaptive learning rates
3. **Convergence**: Tolerance-based stopping criteria
4. **Robustness**: Outlier-resistant estimation methods

### Lomb-Scargle Periodogram

#### Mathematical Foundation

The Lomb-Scargle periodogram computes the power spectrum for unevenly sampled data:

```
P(f) = (1/2σ²) * [Σ(x_i cos(2πf(t_i - τ))]² / Σ cos²(2πf(t_i - τ)) + 
                    [Σ(x_i sin(2πf(t_i - τ))]² / Σ sin²(2πf(t_i - τ))
```

where τ is a time offset that makes the trigonometric functions orthogonal.

#### Implementation Features

1. **Frequency Grid**: Logarithmic spacing for optimal resolution
2. **Peak Detection**: Automatic identification of significant periods
3. **Signal Reconstruction**: Validation using detected periodicities
4. **Performance Metrics**: MSE and MAPE for evaluation

##  Usage Instructions

### Running Linear Regression Models

```bash
cd Project/
python linear_regression_models.py
```

**Expected Output:**
- Model training progress
- Performance metrics comparison
- Convergence plots
- Feature importance analysis

### Running Lomb-Scargle Analysis

```bash
cd Project/
python lomb_scargle_periodogram.py
```

**Expected Output:**
- Periodogram computation
- Peak detection results
- Visualization plots
- Performance evaluation

### Jupyter Notebooks

For interactive analysis, use the provided `.ipynb` files:
- `BE21B032_CH5019_PROJECT_Q1.ipynb` - Linear regression analysis
- `MFDS_Q1.ipynb` - Alternative implementation

## 🔧 Dependencies

### Python Packages
- **Core**: numpy, pandas, scipy
- **ML**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Statistics**: statsmodels

### Data Requirements
- CSV files for insurance and stock data
- Optional: Excel files for additional datasets

## Key Findings

### Robust Regression
1. **LMS and LTS** provide better outlier resistance than OLS
2. **Feature scaling** improves convergence and stability
3. **Gradient clipping** prevents numerical instability in robust methods

### Time Series Analysis
1. **Lomb-Scargle** effectively detects periodicities in unevenly sampled data
2. **Multiple periodicities** can be identified simultaneously
3. **Signal reconstruction** validates periodicity detection quality

## 📚 References

1. **Robust Statistics**: Rousseeuw, P.J. & Leroy, A.M. (1987)
2. **Lomb-Scargle**: Lomb, N.R. (1976), Scargle, J.D. (1982)
3. **Time Series Analysis**: Box, G.E.P., Jenkins, G.M., Reinsel, G.C. (1994)
4. **Course Materials**: Dr. Arun Tangirala, IIT Madras MFDS Course

