# ============================
# Salary Prediction using Linear Regression
# ============================

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================
# Step 1: Load Dataset
# ============================

# Sample dataset (you can upload a CSV from Kaggle with similar columns)
data = {
    'Experience': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    'Education_Level': ['Bachelors', 'Masters', 'Masters', 'PhD', 'Bachelors', 'PhD', 'Masters', 'Bachelors', 'PhD', 'Masters'],
    'Job_Role': ['Data Analyst', 'Data Scientist', 'Software Engineer', 'ML Engineer', 'Data Analyst', 
                 'ML Engineer', 'Software Engineer', 'Data Scientist', 'ML Engineer', 'Data Analyst'],
    'Location': ['Delhi', 'Bangalore', 'Pune', 'Hyderabad', 'Delhi', 'Pune', 'Bangalore', 'Hyderabad', 'Delhi', 'Bangalore'],
    'Salary': [45000, 70000, 90000, 120000, 50000, 135000, 95000, 110000, 125000, 75000]
}

df = pd.DataFrame(data)

# ============================
# Step 2: Data Preprocessing
# ============================

# Encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)

# Split features & target
X = df_encoded.drop('Salary', axis=1)
y = df_encoded['Salary']

# Split into training & testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================
# Step 3: Train Model
# ============================

model = LinearRegression()
model.fit(X_train, y_train)

# ============================
# Step 4: Evaluate Model
# ============================

y_pred = model.predict(X_test)

print("Model Performance:")
print("------------------")
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# ============================
# Step 5: Predict for a New Employee
# ============================

new_data = pd.DataFrame({
    'Experience': [8],
    'Education_Level': ['Masters'],
    'Job_Role': ['Data Scientist'],
    'Location': ['Bangalore']
})

new_data_encoded = pd.get_dummies(new_data)
new_data_encoded = new_data_encoded.reindex(columns=X.columns, fill_value=0)

pred_salary = model.predict(new_data_encoded)
print("\nPredicted Salary for new employee:", int(pred_salary[0]))

# ============================
# Step 6: Visualization
# ============================

plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary (Linear Regression)")
plt.show()
