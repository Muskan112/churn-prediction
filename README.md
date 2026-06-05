# Customer Churn Prediction Model

A machine learning project that predicts which customers are likely to leave a business, built using Python and Scikit-learn.

## What this project does

This model analyses customer data — such as contract type, monthly charges, and tenure — and predicts whether a customer will churn (leave). It achieves approximately 82% accuracy on the Telco Customer Churn dataset.

This kind of predictive model is used by real companies in telecoms, banking, and e-commerce to retain customers before they leave.

## Technologies used

- **Python 3.x**
- **Pandas** — data loading and cleaning
- **Scikit-learn** — machine learning (Random Forest Classifier)
- **Matplotlib** — data visualisation

## Dataset

The project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from IBM via Kaggle (~7,000 customer records).

## How to run it

1. Clone this repository
   ```
   git clone https://github.com/YOUR-USERNAME/churn-prediction.git
   cd churn-prediction
   ```

2. Install required libraries
   ```
   pip install pandas scikit-learn matplotlib
   ```

3. Download the dataset from Kaggle (link above), rename it `telco_churn.csv`, and place it in the project folder

4. Run the model
   ```
   python churn_model.py
   ```

## Results

- **Model accuracy:** ~82%
- **Key finding:** Monthly charges and tenure are the strongest predictors of churn — customers with high bills and short tenure are most at risk
- A feature importance chart is saved automatically as `churn_chart.png`

## Business insight

The model identified that customers on **month-to-month contracts** with **high monthly charges** and **short tenure** are the highest churn risk. This insight could help a business proactively offer discounts or improved service to retain these customers.

## About

Built by Shelly as part of a data analytics portfolio for PhD applications in AI and Business Intelligence.
