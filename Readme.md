# Automated Hyperparameter Optimization using Tree-structured Parzen Estimator (TPE)

This notebook demonstrates automated hyperparameter optimization (HPO) using the Tree-structured Parzen Estimator (TPE) for a RandomForestClassifier on the breast cancer dataset.

## Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Implementation Details](#implementation-details)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Model Training and Optimization](#model-training-and-optimization)
   - [Tree-structured Parzen Estimator (TPE)](#tree-structured-parzen-estimator-tpe)
4. [How to Use](#how-to-use)
5. [Results](#results)
6. [Dependencies](#dependencies)

---

## Introduction

The notebook showcases an automated approach to find the best hyperparameters for a RandomForestClassifier using TPE, a Bayesian optimization technique. It includes data loading, exploratory data analysis (EDA), hyperparameter optimization, model training, and evaluation.

## Dataset

The dataset used is the breast cancer dataset from sklearn (`load_breast_cancer()`). It consists of 569 instances and 30 features, with a binary target variable indicating the presence of breast cancer.

## Implementation Details

### Exploratory Data Analysis (EDA)

- Displays dataset information, summary statistics, class distribution, and missing values.
- Plots histograms, correlation heatmap, boxplots, and distribution plots of features.
- Visualizes the target variable distribution.

### Model Training and Optimization

- Defines functions to create a RandomForestClassifier with specified hyperparameters.
- Implements TPE (Tree-structured Parzen Estimator) from scratch for hyperparameter optimization.
- Defines the objective function for cross-validated ROC AUC score.
- Uses `scipy.stats` for defining parameter space with probability distributions.

### Tree-structured Parzen Estimator (TPE)

The Tree-structured Parzen Estimator (TPE) is a Bayesian optimization technique that models the distribution of good and bad hyperparameters separately. This allows for more efficient search of the hyperparameter space compared to random search or grid search.

#### TPE Implementation

1. **Initialization**: 
   - Start with a specified number of initial random points.
   
2. **Suggestions**:
   - Once the initial points are evaluated, new hyperparameter sets are suggested based on the TPE method.
   - TPE models the likelihood of hyperparameter configurations as coming from one of two Gaussians:
     - One Gaussian models the hyperparameters associated with the top-performing configurations (good configurations).
     - The other models the remaining configurations (bad configurations).
   - The ratio of these likelihoods is used to select new hyperparameter configurations that are more likely to perform well.

3. **Optimization Loop**:
   - Evaluate the suggested hyperparameters and update the Gaussian models.
   - Continue until the specified number of iterations is reached or an early stopping criterion is met.

4. **Output**:
   - The best hyperparameters found during the optimization process.


## How to Use

1. Open the notebook in Google Colab.
2. Run each cell sequentially to load data, perform EDA, optimize hyperparameters, train the model, and evaluate its performance.
3. View the printed results and visualizations to understand the optimization process and model performance.

## Results

The notebook demonstrates how automated hyperparameter optimization using TPE can enhance the performance of a RandomForestClassifier on the breast cancer dataset. Key results include:

- Identification of optimal hyperparameters.
- ROC AUC scores and other classification metrics for model evaluation.
- Comparison of TPE optimization with random search.

## Dependencies

- Python 3.x
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy
