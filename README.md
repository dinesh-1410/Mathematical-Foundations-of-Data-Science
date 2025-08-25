# Mathematical Foundations of Data Science (MFDS) - (CH5019)

**Course:** Mathematical Foundations of Data Science  
**Faculty:** Dr. Arun Tangirala, Wadhwani School of Data Science & AI  
**Institution:** IIT Madras  
**Duration:** January 2024 - May 2024  
**Student:** BE21B032

## 🎯 Course Overview

This repository contains comprehensive implementations and solutions for the Mathematical Foundations of Data Science course, covering essential mathematical concepts and their practical applications in data science.

## 📚 Course Content

### Core Topics Covered
- **Linear Algebra Fundamentals**
- **Probability & Statistics**
- **Optimization Theory**
- **Parameter Estimation**
- **System Identification**
- **Time Series Analysis**
- **Robust Estimation Methods**

### Mathematical Concepts
- Ordinary Least Squares (OLS)
- Least Median Squares (LMS)
- Least Trimmed Squares (LTS)
- Lomb-Scargle Periodogram
- ARIMA Models
- Robust Regression
- Principal Component Analysis (PCA)

## 🚀 Key Projects

### 1. Linear Regression Models Implementation
- **File:** `Project/linear_regression_models.py`
- **Description:** Comprehensive implementation of OLS, LMS, and LTS regression methods
- **Application:** Insurance analytics with evaluation using MSE, RB, and MAE metrics
- **Features:**
  - Custom gradient descent implementation
  - Robust estimation methods
  - Performance benchmarking

### 2. Lomb-Scargle Periodogram Analysis
- **File:** `Project/lomb_scargle_periodogram.py`
- **Description:** Python implementation of Lomb-Scargle periodogram for time series analysis
- **Application:** Tesla stock price data analysis
- **Benchmarking:** Comparison with ARIMA models using NMSE and MAPE metrics

### 3. Insurance Analytics Project
- **Dataset:** `Project/medical_insurance.csv`
- **Analysis:** Comprehensive insurance data analysis using robust regression methods
- **Evaluation Metrics:** MSE, Relative Bias (RB), Mean Absolute Error (MAE)

## 📁Repository Structure

```
Mathematical-Foundations-of-Data-Science/
├── Project/                 # Main course project files
│   ├── linear_regression_models.py
│   ├── lomb_scargle_periodogram.py
│   ├── medical_insurance.csv
│   ├── Tesla Stock Price.csv
│   └── README.md
├── Assignments/            # Course assignments
│   ├── A_1/               # Assignment 1
│   ├── A_2/               # Assignment 2
│   ├── A_3/               # Assignment 3
│   ├── A_4/               # Assignment 4
│   ├── A_5/               # Assignment 5
│   └── README.md
├── Tutorial/               # Tutorial solutions
├── Notes/                  # Course notes and materials
├── SLIDES/                 # Lecture slides
├── Textbooks/              # Reference materials
├── requirements.txt        # Python dependencies
├── .gitignore             # Git exclusions
└── README.md              # Main repository documentation
```

### Programming Languages
- **Python 3.8+:** Primary language for data analysis and machine learning
- **MATLAB R2020b+:** Signal processing and advanced mathematical computations

### Python Libraries
- **Core Scientific Computing:**
  - NumPy (≥1.21.0) - Numerical computations and array operations
  - Pandas (≥1.3.0) - Data manipulation and analysis
  - SciPy (≥1.7.0) - Scientific computing and optimization

- **Machine Learning & Statistics:**
  - Scikit-learn (≥1.0.0) - Machine learning algorithms and metrics
  - Statsmodels (≥0.13.0) - Advanced statistical analysis

- **Data Visualization:**
  - Matplotlib (≥3.4.0) - Static plotting and visualization
  - Seaborn (≥0.11.0) - Statistical data visualization
  - Plotly (≥5.0.0) - Interactive plotting

- **Development Environment:**
  - Jupyter (≥1.0.0) - Interactive notebooks
  - IPython Kernel (≥6.0.0) - Enhanced Python shell


### MATLAB Toolboxes
- **Signal Processing Toolbox** - Lomb-Scargle periodogram implementation
- **Statistics and Machine Learning Toolbox** - Statistical analysis

## Academic Context

This course provides the mathematical foundation necessary for advanced data science applications, covering:
- Statistical estimation theory
- Robust methods for real-world data
- Time series analysis techniques
- Optimization in parameter estimation

## 📝 Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Mathematical-Foundations-of-Data-Science.git
   cd Mathematical-Foundations-of-Data-Science
   ```

2. **Run Python implementations:**
   ```bash
   python Project/linear_regression_models.py
   ```

3. **Open MATLAB files** in MATLAB or Octave for periodogram analysis



