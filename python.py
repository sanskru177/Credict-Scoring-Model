import pandas as pd

# Column names based on UCI description
columns = [
    'status', 'duration', 'credit_history', 'purpose', 'amount', 'savings',
    'employment_duration', 'installment_rate', 'personal_status_sex',
    'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
    'housing', 'number_credits', 'job', 'people_liable', 'telephone',
    'foreign_worker', 'target'
]

# Load the raw dataset (downloaded as german.data)
df = pd.read_csv('german.data', sep=' ', header=None)
df.columns = columns

# Convert the target values: 1 = Good, 2 = Bad
df['target'] = df['target'].map({1: 1, 2: 0})

# Save as a clean CSV
df.to_csv('credit_data.csv', index=False)

print("✅ credit_data.csv has been saved successfully and is ready to use!")
