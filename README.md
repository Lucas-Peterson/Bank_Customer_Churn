# Bank Customer Churn Prediction

This project demonstrates an end-to-end process for predicting customer churn in a retail bank using Python and machine learning. The workflow covers data preprocessing, exploratory analysis, model building, evaluation, and business recommendations.

## Overview

- **Goal:** Identify which bank customers are at high risk of churning (leaving the bank), and understand key factors driving churn.
- **Approach:** We use real customer data, perform feature engineering and EDA, train several classification models, and interpret the results using visualizations and model explanations.
- **Main Steps:**
  1. Data Loading and Exploration
  2. Data Preprocessing
  3. Feature Engineering
  4. Model Selection and Evaluation
  5. Model Interpretation
  6. Recommendations

## Dataset

The dataset contains anonymized customer data, including demographics, account information, and churn labels.  
**Source:** [Kaggle - Churn Bank Customer](https://www.kaggle.com/datasets/kartiksaini18/churn-bank-customer/data)

**Columns include:**
- CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited (churn label)

## Project Structure

- `Bank_Customer_Churn_Prediction.ipynb`: Jupyter notebook with the full workflow and analysis.
- `Churn_Modelling.csv`: The input dataset (not provided here for copyright reasons).
- Image files: Visualizations used in the notebook (boxplots, feature importances, ROC curves, etc.).

## Example Analysis

- **Boxplots and KDE plots** are used for EDA to reveal how age and number of products relate to churn.
- **Feature importance** (for Gradient Boosting) shows that Age, NumOfProducts, and IsActiveMember are key drivers of churn.
- **ROC curves** and classification reports compare different models, with Gradient Boosting achieving an AUC of 0.87.

## Key Results

- **Older customers** and those with only 1 or more than 2 products are most likely to churn.
- **Gradient Boosting** outperforms Logistic Regression and Random Forest in distinguishing churners from loyal customers.
- **Model insights** can be used to design targeted retention strategies.

## How to Run

1. Install dependencies:
    ```bash
    pip install pandas scikit-learn matplotlib seaborn
    ```
2. Dataset ist aleady imported, but you can download it.
3. Open `Bank_Customer_Churn_Prediction.ipynb` in Jupyter Notebook or JupyterLab.
4. Run the notebook cell by cell to see the analysis and results.

## Recommendations

- **Target high-risk groups:** Proactively engage older clients and those with suboptimal product portfolios.
- **Increase product engagement:** Promote holding exactly two products as this segment has the lowest churn.
- **Integrate model into business process:** Use the provided churn risk scores to trigger retention actions in real time.
- **Further improvements:** Enrich the dataset with behavioral, transactional, or satisfaction data to improve prediction and deepen business understanding.

## License and Dataset

This project is provided for educational purposes only.  
Dataset license: [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
[Dataset](https://www.kaggle.com/datasets/kartiksaini18/churn-bank-customer/data)
