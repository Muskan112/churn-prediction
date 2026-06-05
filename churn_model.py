# ============================================================
#  Customer Churn Prediction Model
#  Author: Shelly
#  Description: Predicts which customers are likely to leave
#                a business using machine learning.
#  Libraries: pandas, scikit-learn, matplotlib
# ============================================================


# ---- STEP 1: Import libraries ----
# Think of these as "tools" we borrow to do the work

import pandas as pd                              # For loading and organising data (like Excel)
import matplotlib.pyplot as plt                  # For drawing charts
from sklearn.model_selection import train_test_split   # For splitting data into train/test
from sklearn.preprocessing import LabelEncoder         # For converting text (Yes/No) to numbers
from sklearn.ensemble import RandomForestClassifier    # The ML algorithm (our "brain")
from sklearn.metrics import accuracy_score, classification_report  # For checking how good our model is


# ---- STEP 2: Load the dataset ----
# Make sure telco_churn.csv is in the same folder as this file

print("Loading data...")
df = pd.read_csv('telco_churn.csv')              # Read the CSV file into a table called df

print(f"Dataset loaded: {df.shape[0]} rows and {df.shape[1]} columns")  # Show how big the data is


# ---- STEP 3: Clean the data ----
# Real-world data is messy — we need to fix it before using it

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# 'TotalCharges' column has some spaces instead of numbers — this converts them properly
# errors='coerce' means: if something can't be converted, turn it into NaN (empty)

df = df.dropna()
# Remove any rows that still have empty/missing values
# Machine learning models cannot handle empty cells

print(f"After cleaning: {df.shape[0]} rows remaining")


# ---- STEP 4: Convert text columns to numbers ----
# ML models only understand numbers, not words like "Yes", "No", "Male", "Female"

le = LabelEncoder()                              # Create a label encoder tool

# Convert the target column (what we want to predict)
df['Churn'] = le.fit_transform(df['Churn'])
# fit_transform learns the mapping (Yes=1, No=0) and applies it
# Now "Yes" becomes 1 and "No" becomes 0

# Convert other text columns that the model will use
df['gender'] = le.fit_transform(df['gender'])
df['Contract'] = le.fit_transform(df['Contract'])
df['InternetService'] = le.fit_transform(df['InternetService'])
df['PaymentMethod'] = le.fit_transform(df['PaymentMethod'])


# ---- STEP 5: Choose features (inputs) and target (output) ----
# Features = the information we give the model to learn from
# Target = the answer we want the model to predict

features = [
    'tenure',           # How many months the customer has been with the company
    'MonthlyCharges',   # How much they pay per month
    'TotalCharges',     # Total amount they have paid overall
    'gender',           # Customer gender (0 or 1 after encoding)
    'Contract',         # Type of contract (month-to-month, 1 year, 2 year)
    'InternetService',  # Type of internet service
    'PaymentMethod'     # How they pay their bills
]

X = df[features]        # X = all the input columns (features)
y = df['Churn']         # y = the output column we want to predict (churn yes/no)

print(f"\nFeatures used for prediction: {features}")
print(f"Total samples: {len(X)}")


# ---- STEP 6: Split data into training and testing sets ----
# We train on 80% of the data, then test on the remaining 20%
# This is like studying with 80% of practice questions, then being tested on unseen ones

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% goes to testing
    random_state=42      # This number makes results reproducible — same result every run
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ---- STEP 7: Create and train the model ----
# Random Forest = 100 decision trees working together
# Like asking 100 experts and going with the majority vote

print("\nTraining the model... (this takes a few seconds)")

model = RandomForestClassifier(
    n_estimators=100,    # Use 100 decision trees
    random_state=42      # Makes results reproducible
)

model.fit(X_train, y_train)
# .fit() is where the learning happens
# The model reads all training data and finds patterns
# e.g. "customers with high monthly charges AND short tenure tend to churn"

print("Model training complete!")


# ---- STEP 8: Make predictions ----
# Now we test the model on data it has NEVER seen before

y_pred = model.predict(X_test)
# The model looks at each test customer's features
# and predicts: will they churn (1) or stay (0)?


# ---- STEP 9: Evaluate how good the model is ----

accuracy = accuracy_score(y_test, y_pred)
# Compare predictions vs the real answers
# If the model predicted 82 out of 100 correctly, accuracy = 82%

print(f"\n{'='*45}")
print(f"  MODEL ACCURACY: {accuracy:.2%}")
print(f"{'='*45}")
print("\nDetailed report:")
print(classification_report(y_test, y_pred, target_names=['Stayed', 'Churned']))
# This shows precision, recall, and F1-score for each class
# Precision = of all predicted churners, how many actually churned?
# Recall    = of all actual churners, how many did we catch?


# ---- STEP 10: Visualise which features matter most ----
# Feature importance = which columns had the biggest effect on the prediction

print("\nGenerating feature importance chart...")

importances = model.feature_importances_
# This gives a score for each feature — higher = more important

# Sort features by importance (highest first)
sorted_idx = importances.argsort()

plt.figure(figsize=(9, 5))
plt.barh(
    [features[i] for i in sorted_idx],   # Feature names on y-axis
    importances[sorted_idx],              # Importance scores on x-axis
    color='#378ADD',                      # Blue bars
    edgecolor='none'
)
plt.title('Which customer factors predict churn most strongly?', fontsize=13, pad=12)
plt.xlabel('Importance score (higher = stronger predictor)', fontsize=11)
plt.tight_layout()
plt.savefig('churn_chart.png', dpi=150, bbox_inches='tight')
# Saves the chart as an image in your project folder
print("Chart saved as 'churn_chart.png'")
plt.show()
# Opens the chart on your screen


# ---- STEP 11: Predict a single new customer ----
# This shows how the model would work in a real business setting

print("\n--- Predicting for a new customer ---")

new_customer = pd.DataFrame([{
    'tenure': 2,              # Only been with company 2 months
    'MonthlyCharges': 85.0,   # Pays £85/month (high)
    'TotalCharges': 170.0,    # Paid £170 total so far
    'gender': 1,              # Female (1 after encoding)
    'Contract': 0,            # Month-to-month contract (risky!)
    'InternetService': 1,     # Has fibre optic
    'PaymentMethod': 2        # Pays by electronic check
}])

prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)[0][1]
# predict_proba gives the probability (0 to 1) that this customer will churn

if prediction[0] == 1:
    print(f"Prediction: This customer is LIKELY TO CHURN")
else:
    print(f"Prediction: This customer is LIKELY TO STAY")

print(f"Churn probability: {probability:.1%}")
print("\nDone! Your model is working successfully.")
