Used Bikes Data Analysis and Price Prediction
Project Overview
This project focuses on analyzing a dataset of used motorcycles (used_bikes.csv) to identify factors influencing their market prices and to develop a machine learning model for accurate price prediction. The dataset includes attributes such as model name, model year, kilometers driven, ownership status, location, mileage, power, and price. The project encompasses data loading, cleaning, exploratory data analysis (EDA), feature engineering, model training, and hyperparameter tuning, with the tuned XGBoost model achieving superior predictive performance.
Objectives

Load and preprocess the used bikes dataset.
Conduct EDA to uncover patterns and correlations affecting bike prices.
Engineer features to enhance model performance.
Train and evaluate machine learning models, optimizing the best model for price prediction.
Provide actionable insights for stakeholders in the used bike market.

Dataset
The dataset (used_bikes.csv) contains the following columns:

model_name: Name and model of the bike (e.g., Bajaj Avenger Cruise 220 2017).
model_year: Year of manufacture.
kms_driven: Total kilometers driven.
owner: Ownership status (e.g., first owner).
location: City of listing.
mileage: Fuel efficiency (kmpl or kms).
power: Engine power (bhp).
price: Sale price (target variable, in INR).

Methodology

Data Loading:

Loaded dataset into a Pandas DataFrame using pd.read_csv().
Displayed first 5 rows to verify successful loading.


Data Cleaning:

Handled missing values: Dropped rows with missing location (19), mileage (11), and power (31) values.
Cleaned numerical columns (price, kms_driven, mileage, power) by removing non-numeric characters and converting to numeric types.
Capped outliers at the 99th percentile.
Removed duplicate rows to ensure data quality.


Feature Engineering:

Created brand feature from model_name.
Derived age feature from model_year.
One-hot encoded categorical variables (brand, location, owner).


Exploratory Data Analysis (EDA):

Identified strong positive correlation between power and price.
Found strong negative correlation between mileage and price.
Visualized distributions using histograms and box plots, revealing right-skewed price and outliers.
Feature importance analysis highlighted power, mileage, and brand as key predictors.


Data Preparation:

Split data: 80% training, 20% testing.
Scaled numerical features using StandardScaler.


Model Building:

Evaluated models: XGBoost and Linear Regression (default parameters).
Default XGBoost: R² ~0.78, MAE ~5000, RMSE ~7500 (best performer).
Linear Regression: Negative R² (poor fit due to non-linear relationships).
Optimized XGBoost using GridSearchCV, improving performance:
R² ~0.85, MAE ~4500, RMSE ~6800.




Model Evaluation:

Visualized tuned XGBoost performance:
Actual vs. Predicted Prices: Strong correlation.
Residual Plot: Random residuals, indicating no systematic bias.
Feature Importance: power, mileage, brand as top predictors.





Key Findings

The tuned XGBoost model explained approximately 85% of the variance in bike prices, with low MAE (4500 INR) and RMSE (6800 INR).
Power and mileage are the strongest predictors of price, with brand and model_name also significant.
EDA revealed a right-skewed price distribution and outliers, addressed through preprocessing.
The model provides a reliable tool for stakeholders to estimate used bike prices based on key attributes.

Next Steps

Explore advanced feature engineering (e.g., interaction terms).
Test additional models like LightGBM or CatBoost.
Implement ensemble techniques for improved accuracy.
Acquire a larger, more diverse dataset.
Apply rigorous cross-validation and feature selection methods.

Repository Structure

used_bikes.csv: Dataset of used motorcycles.
bike.ipynb: Jupyter notebook containing data analysis and model building.
README.md: Project documentation (this file).

Setup Instructions

Clone the Repository:git clone https://github.com/[your-username]/[your-repo-name].git


Install Dependencies:Ensure Python 3.x is installed. Install required packages:pip install pandas numpy scikit-learn xgboost matplotlib seaborn


Run the Notebook:
Open bike.ipynb in Jupyter Notebook or JupyterLab.
Ensure used_bikes.csv is in the same directory.
Execute cells sequentially to reproduce the analysis.


Environment:
Compatible with Google Colab or local Jupyter environments.
Recommended: Use a virtual environment to manage dependencies.



Requirements

Python 3.x
Libraries: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

Results

Best Model: Tuned XGBoost
R²: ~0.85
MAE: ~4500 INR
RMSE: ~6800 INR


Visualizations confirm robust predictions with no systematic bias.
Feature importance emphasizes power, mileage, and brand.

Acknowledgments

Dataset sourced from [specify source if known, e.g., Kaggle, or note as provided].
Built using Python, Pandas, Scikit-learn, and XGBoost.

License
This project is licensed under the MIT License. See the LICENSE file for details.
