# Gold-Price-Prediction
This project aims to predict gold prices using a Random Forest Regressor machine learning model. The model is trained on historical gold price data, and predictions are made on new, unseen data.


## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Data Collection and Processing](#data-collection-and-processing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)


## Overview

The project includes Python code that performs the following tasks:

- Imports necessary libraries (pandas, numpy, matplotlib, seaborn, scikit-learn).
- Collects and processes historical gold price data from a CSV file.
- Conducts exploratory data analysis (EDA) to understand the dataset.
- Trains a Random Forest Regressor on the features and target variable.
- Evaluates the model's performance using the R-squared error metric.
- Visualizes the actual vs predicted gold prices using matplotlib.

## Prerequisites

- Python (version 3.11.7)
- Required Python packages: pandas, numpy, matplotlib, seaborn, scikit-learn

### Installing Dependencies

```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/heyismail/Gold-Price-Prediction.git
cd Gold-Price-Prediction
```

2. Set up a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Collection and Processing

- The dataset is collected from a CSV file using pandas.
- Insights, statistical descriptions, and correlations are examined.
- Date columns are converted to datetime format.

## Model Training and Evaluation

- Features and target variable are split.
- Random Forest Regressor is trained with 100 decision trees.
- Model evaluation is performed using R-squared error.

## Results

- Predictions are made on test data.
- Actual vs predicted gold prices are visualized using matplotlib.
