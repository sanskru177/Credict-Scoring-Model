# credit_scoring_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import joblib

# -----------------------------------------
# 1. Load Dataset
# -----------------------------------------
df = pd.read_csv("credit_data.csv") 
print("✅ Data loaded successfully!")
print(df.head())

# -----------------------------------------
# 2. Data Preprocessing
# -----------------------------------------

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split into features and target
X = df.drop('target', axis=1)
y = df['target']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------
# 3. Train-Test Split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------------------
# 4. Train Models
# -----------------------------------------

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# -----------------------------------------
# 5. Evaluate Models
# -----------------------------------------

# Logistic Regression
lr_preds = lr_model.predict(X_test)
lr_probs = lr_model.predict_proba(X_test)[:, 1]
print("\n📊 Logistic Regression Report:")
print(classification_report(y_test, lr_preds))
print("ROC-AUC Score:", roc_auc_score(y_test, lr_probs))

# Random Forest
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]
print("\n🌲 Random Forest Report:")
print(classification_report(y_test, rf_preds))
print("ROC-AUC Score:", roc_auc_score(y_test, rf_probs))

# -----------------------------------------
# 6. Plot ROC Curve
# -----------------------------------------
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)

plt.figure(figsize=(8, 5))
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression (AUC = {:.2f})".format(roc_auc_score(y_test, lr_probs)))
plt.plot(fpr_rf, tpr_rf, label="Random Forest (AUC = {:.2f})".format(roc_auc_score(y_test, rf_probs)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()
plt.show()

# -----------------------------------------
# 7. Save the Models and Scaler
# -----------------------------------------
joblib.dump(lr_model, "logistic_model.pkl")
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Models and scaler saved successfully!")
