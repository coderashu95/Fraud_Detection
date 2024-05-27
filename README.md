# Fraud Detection in Insurance Claims

## Project Overview

This project aims to develop a robust machine learning model to predict fraudulent insurance claims. Accurately detecting fraud can significantly reduce financial losses for the company and improve the overall efficiency of the claims process. The dataset used for this project contains various features related to insurance claims, and the target variable indicates whether a claim is fraudulent.

## Business Case

Insurance fraud is a critical issue in the industry, leading to substantial financial losses each year. By leveraging machine learning techniques, we aim to identify potentially fraudulent claims early in the process, thereby saving costs and improving customer satisfaction by minimizing false accusations.

## Tools and Libraries

The following tools and libraries were used in this project:

- **Python Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost
- **Data Processing:** pandas, numpy
- **Data Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn, xgboost

## Approach

### Data Preprocessing

- **Loading Data:** The dataset was loaded using pandas.
- **Exploratory Data Analysis (EDA):** Visualizations were created using seaborn and matplotlib to understand the distribution of features and the target variable.
- **Handling Missing Values:** Missing values were appropriately handled to ensure data quality.
- **Encoding Categorical Variables:** Categorical variables were encoded using one-hot encoding.
- **Feature Scaling:** Numerical features were scaled to ensure uniformity.

### Model Building

Several machine learning models were tested to identify the best-performing model:

1. **k-Nearest Neighbors (kNN)**
   - Hyperparameter tuning was performed to find the best value of 'k'.
   - Performance was evaluated using various metrics.

2. **Logistic Regression**
   - A logistic regression model was trained and evaluated.
   - Performance metrics were computed for different probability thresholds.

3. **Random Forest Classifier**
   - A Random Forest model was built with 200 estimators.
   - Cross-validation was used to evaluate the model's performance.

4. **XGBoost Classifier**
   - An XGBoost model was trained with hyperparameter tuning.
   - GridSearchCV was used to find the best hyperparameters.

### Model Evaluation

The models were evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Log Loss**
- **Brier Score**

### Recommended Model

After careful evaluation, the XGBoost Classifier was chosen as the recommended model due to its superior performance across multiple metrics. XGBoost showed higher accuracy, better precision, recall, and lower log loss compared to other models. Additionally, the hyperparameter tuning process further enhanced its predictive capability.

## Conclusion

The XGBoost Classifier was selected as the final model for predicting fraudulent insurance claims. This model not only demonstrated the best performance during evaluation but also provides flexibility for further tuning and improvement. Implementing this model can help the insurance company effectively identify and mitigate fraudulent claims, leading to significant cost savings and operational efficiency.

## How to Use

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
