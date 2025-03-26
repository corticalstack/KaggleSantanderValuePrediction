# 🚀 Santander Customer Value Prediction

A machine learning solution for the Kaggle Santander Customer Value Prediction competition, achieving high accuracy through model stacking and ensemble techniques.

## 📚 Description

This repository contains a solution for the [Santander Customer Value Prediction](https://www.kaggle.com/c/santander-value-prediction-challenge) Kaggle competition. The challenge involves predicting the value of transactions for potential customers, helping Santander provide a more personalized customer experience.

The solution implements a stacked ensemble approach combining:
- Gradient Boosting Regressor
- LightGBM Regressor
- Random Forest Regressor

These models are combined using a Lasso regression stacking technique to produce the final predictions, with k-fold cross-validation to ensure robustness.

## 🔧 Prerequisites

- Python 3.x
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - lightgbm

## 📊 Features

- **Feature Selection**: Implements a curated list of the most predictive features
- **Data Preprocessing**: Handles missing values and adds statistical features
- **Model Ensemble**: Combines multiple regression models for improved accuracy
- **K-Fold Validation**: Uses 20-fold cross-validation for robust performance evaluation
- **Stacking Technique**: Employs model stacking with Lasso regression for final predictions

## 🛠️ Setup Guide

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/KaggleSantanderValuePrediction.git
   cd KaggleSantanderValuePrediction
   ```

2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn lightgbm
   ```

3. Download the competition data from [Kaggle](https://www.kaggle.com/c/santander-value-prediction-challenge/data) and place the `train.csv` and `test.csv` files in the repository root directory.

## 🔬 Usage

Run the main script to train the models and generate predictions:

```bash
python valuePrediction.py
```

This will:
1. Load and preprocess the training and test data
2. Train multiple regression models using k-fold cross-validation
3. Create an ensemble prediction using model stacking
4. Generate a `test_set_prediction.csv` file with predictions in the format required for Kaggle submission

## 🧠 Approach

The solution follows these key steps:

1. **Data Preprocessing**:
   - Removing columns with only zeros
   - Imputing missing values
   - Adding statistical features (mean, median, sum, standard deviation, kurtosis)

2. **Feature Selection**:
   - Using a predefined list of the most predictive features

3. **Model Training**:
   - Training multiple regression models with optimized hyperparameters
   - Using 20-fold cross-validation to ensure robustness

4. **Ensemble Creation**:
   - Combining model predictions using stacking
   - Using Lasso regression as a meta-learner
   - Applying geometric mean for final prediction aggregation

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
