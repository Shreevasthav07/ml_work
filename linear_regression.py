
# PAMPATTIWAR SHREEVASTHAV [25MSCSCPY0003]
# M.Sc. COMPUTER SCIENCE [DATA ANALYTICS]

#===================================================================
# SAMPLE CODE TO IMPLEMENT LINEAR REGRESSION
#===================================================================

# ====== STEP 1 : IMPORT REQUIRED LIBRARIES =======================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#===================================================================

# ========= STEP 2 : LOAD THE DATASET INTO A DATAFRAME =============
df = pd.read_csv("salary_data.csv")
print("First 5 Rows: ")
print(df.head())

print("\n Dataset Info:")
print(df.info())

print("\n Statistical Summary: ")
print(df.describe())
#==================================================================

#========== STEP 3 : DATA CLEANING =================================
print("\nMissing Values in each Column:")
print(df.isnull().sum())
    # drop missing values or replace it with mean
    # df = df.dropna() or df.fillna(df.mean(), inplace = True)
df = df.dropna()
    # drop duplicate records
df = df.drop_duplicates()
#=================================================================

#============== STEP 4 : FEATURE SELECTION ========================
X = df[['YearsExperience']]
y = df['Salary']
#================================================================

#============== STEP 5 : TRAIN TEST SPLIT ==============================================
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
#=======================================================================================

#============== STEP 6 : STANDARDIZATION ===============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#========================================================================================

#============== STEP 7 : MODEL TRAINING =================================================
model = LinearRegression()
model.fit(X_train_scaled,y_train)

print("\n Model Trained Successfully!")
print("Coefficent (Slope): ", model.coef_)
print("Intercept: ", model.intercept_)
#========================================================================================

#============== STEP 8 : PREDICTIONS =====================================================
y_pred = model.predict(X_test_scaled)
#=========================================================================================

#============== STEP 9 : EVALUATION ======================================================
mse = mean_squared_error(y_test, y_pred) # (1/n) * summation((Yi - Yi^)**2)
rmse = np.sqrt(mse)  #sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation: ")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R2 Score: {r2:.3f}")

#============== STEP 10 : VISUALIZATIONS ================================================

#SCATTER PLOT : YEARSEXPERIENCE VS SALARY
plt.figure(figsize=(8,6))
plt.scatter(df['YearsExperience'], df['Salary'], color='blue')
plt.title("Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.grid(True)
plt.show()

# Regression Line (training data)

plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train, color='blue', label="Training Data")
plt.plot(X_train, model.predict(scaler.transform(X_train)), color='red', label="Regression Line")
plt.title("Regression Line (Training Data)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.show()

# Actual vs Predicted (Testing data)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='purple')
plt.plot(y_test, y_test, color='red') # Perfect prediction line
plt.title("Actual vs Predicted Salaries")
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.grid(True)
plt.show()

#======================= THE - END ====================================================









