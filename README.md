MATLAB Marketing Analytics

📊 A MATLAB-based project for cleaning, analyzing, and modeling marketing data.
This project simulates a real-world dataset of Ad Spend vs. Website Traffic and demonstrates an end-to-end data analysis workflow in MATLAB.

🚀 Project Overview

This project was built to showcase practical MATLAB skills in data analysis, visualization, anomaly detection, and regression modeling.

Key objectives:

Import and clean raw marketing data (CSV).

Handle missing values and outliers.

Engineer new features such as moving averages and weekend effects.

Explore the data visually (time series, scatter, boxplot).

Build a baseline regression model to predict website visits from ad spend.

Apply a robust regression to reduce the influence of anomalies.

🛠 Features

Data Cleaning → Linear interpolation for missing values, removal of impossible entries.

Exploratory Data Analysis (EDA) → Plots for time series, scatter trends, and weekly effects.

Anomaly Detection → Median Absolute Deviation (MAD) for outlier flagging.

Modeling → Ordinary Least Squares (OLS) and weighted robust regression.

Visualization → Actual vs. Predicted visits, highlighting anomalies.

Reproducibility → Pipeline from raw CSV → cleaned dataset → plots → regression results.

📂 Repository Structure
matlab-marketing-analytics/
│
├── marketing_data_analysis.m     % Main MATLAB script (run this file)
├── marketing_data.csv             % Synthetic raw dataset (auto-generated if missing)
├── marketing_data_cleaned.csv     % Cleaned dataset with new features
└── README.md                      % Project documentation

▶️ How to Run

Clone the repository:

git clone https://github.com/<your-username>/matlab-marketing-analytics.git
cd matlab-marketing-analytics


Open MATLAB and run:

marketing_data_analysis


The script will:

Generate a synthetic dataset (if not already provided).

Clean and enrich the dataset.

Produce visualizations.

Print regression model metrics (R², MAE, RMSE).

Save the cleaned dataset as marketing_data_cleaned.csv.
