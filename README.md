# UCI Credit Card Default Prediction

## Overview

This project focuses on predicting credit card defaults using machine learning techniques. The dataset used is the UCI Credit Card dataset, which contains information about credit card holders and their payment behaviors. The goal is to build models that can accurately predict whether a customer will default on their next payment.

## Dataset

The dataset contains the following features:

- **LIMIT_BAL**: Credit limit
- **SEX**: Gender (1 = male, 2 = female)
- **EDUCATION**: Education level
- **MARRIAGE**: Marital status
- **AGE**: Age in years
- **PAY_0 to PAY_6**: History of past payments
- **BILL_AMT1 to BILL_AMT6**: Amount of bill statements
- **PAY_AMT1 to PAY_AMT6**: Amount of previous payments
- **default**: Target variable (1 = default, 0 = no default)

## Data Preprocessing

1. **Data Loading**: The dataset is loaded using pandas.
2. **Feature Renaming**: The target column is renamed from "default.payment.next.month" to "default" for simplicity.
3. **Missing Values**: The dataset is checked for missing values, and none are found.
4. **Feature Scaling**: Features are standardized using `StandardScaler` to ensure all features contribute equally to the model.
5. **Train-Test Split**: The data is split into training (80%) and testing (20%) sets, stratified by the target variable to maintain class distribution.

## Models

Two machine learning models are implemented and evaluated:

### 1. Logistic Regression
- **Description**: A linear model for binary classification.
- **Results**:
  - Precision: 0.82 (non-default), 0.69 (default)
  - Recall: 0.97 (non-default), 0.24 (default)
  - F1-score: 0.89 (non-default), 0.36 (default)
  - ROC-AUC Score: 0.707

### 2. Decision Tree Classifier
- **Description**: A tree-based model with a maximum depth of 5.
- **Results**:
  - Precision: 0.84 (non-default), 0.66 (default)
  - Recall: 0.95 (non-default), 0.36 (default)
  - F1-score: 0.89 (non-default), 0.46 (default)
  - ROC-AUC Score: 0.742

## Evaluation

The models are evaluated using:
- **Classification Report**: Provides precision, recall, and F1-score for each class.
- **Confusion Matrix**: Shows the number of true positives, false positives, true negatives, and false negatives.
- **ROC-AUC Score**: Measures the area under the receiver operating characteristic curve, indicating the model's ability to distinguish between classes.

## Results

- The Logistic Regression model performs well in predicting non-default cases but struggles with defaults, as indicated by the low recall for the default class.
- The Decision Tree model shows a slight improvement in predicting defaults, with a higher recall and ROC-AUC score compared to Logistic Regression.

## How to Run the Code

1. **Prerequisites**:
   - Python 3.x
   - Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

2. **Steps**:
   - Clone the repository or download the Jupyter notebook.
   - Ensure the dataset file (`UCI_Credit_Card.csv`) is in the correct path.
   - Run the notebook cells sequentially to preprocess the data, train the models, and evaluate the results.

## Future Work

- **Feature Engineering**: Explore additional features or transformations to improve model performance.
- **Model Tuning**: Experiment with hyperparameter tuning for better results.
- **Advanced Models**: Try ensemble methods like Random Forest or Gradient Boosting to enhance prediction accuracy.
- **Imbalanced Data Handling**: Address class imbalance using techniques like SMOTE or class weights.

## Conclusion

This project demonstrates the application of machine learning to predict credit card defaults. The Decision Tree model shows promising results, but further improvements can be made to enhance the prediction of default cases. The insights gained can help financial institutions mitigate risks by identifying potential defaulters early.
