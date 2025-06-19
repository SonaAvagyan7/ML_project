# ML_project_water_quality_prediction
A water potability prediction project using machine learning models, enhancing accuracy via data preprocessing and optimized Random Forest.

Water Quality Prediction
========================

This project uses machine learning models to predict whether water is safe to drink based on its physical and chemical properties.

Project Overview
----------------

Using supervised classification algorithms, this notebook aims to automate water safety classification based on measurable features. It includes data exploration, preprocessing, model training, and evaluation.

Dataset
-------

- Source: Kaggle - Water Potability Dataset
- Observations: 3,276 samples
- Features:
  - pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity
  - Target: Potability (0 = Not potable, 1 = Potable)

Data Preprocessing
------------------

- Handled missing values using SimpleImputer (mean strategy)
- Scaled features using StandardScaler
- Splitted dataset into training and testing sets
- No categorical variables

Exploratory Data Analysis
-------------------------

- Correlation matrix to identify relationships
- Distribution plots to inspect feature behavior
- Visual inspection of class imbalance in potability

Models Used
-----------

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

Evaluation Metrics
------------------

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Cross-validation with 5 folds

Results Summary
---------------

| Model               | Accuracy | ROC-AUC |
|--------------------|----------|---------|
| Logistic Regression| 0.609756| 0.548467  |
| Random Forest      | 0.673780| 0.645015  |
| Gradient Boosting  | 0.652439| 0.627300  |
| XGBoost            | 0.647866| 0.625449  |


Technical Details
-----------------

- Language: Python 3.8+
- Notebook: Jupyter
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost


Business Value
--------------

- Helps identify safe drinking water automatically
- Supports real-time field testing in remote areas
- Enhances public health monitoring and policy-making

Conclusion
----------

Machine learning models can effectively predict water potability. With proper feature processing and model selection, reliable automated classification is possible, with potential real-world applications.
