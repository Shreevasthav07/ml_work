import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB

# Load dataset
df = pd.read_csv("play.csv")

# Drop serial-number column
df = df.drop(columns=["Day"])

target = "PlayTennis"

# Store encoders for each column
encoders = {}

# Encode all remaining columns
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Split features & label
X = df.drop(columns=[target])
y = df[target]

# Train Naive Bayes
model = CategoricalNB()
model.fit(X, y)

print("Accuracy:", model.score(X, y))

# Prediction sample
sample = {
    "Outlook": "Sunny",
    "Temp": "Cool",
    "Humidity": "High",
    "Wind": "Strong"
}

# Encode sample
sample_encoded = [
    [encoders[col].transform([sample[col]])[0] for col in X.columns]
]

# Create DataFrame
sample_df = pd.DataFrame(sample_encoded, columns=X.columns)

# Predict (numeric)
prediction = model.predict(sample_df)[0]

# Convert numeric → Yes/No
predicted_label = encoders[target].inverse_transform([prediction])[0]

print("Prediction:", predicted_label)