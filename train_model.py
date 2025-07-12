import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = pd.read_csv("data.csv")
df["Skills"] = df["Skills"].str.split(", ")  # Convert skills to lists

# Encode categorical features
mlb = MultiLabelBinarizer()
skills_encoded = pd.DataFrame(mlb.fit_transform(df["Skills"]), columns=mlb.classes_)

education_encoded = pd.get_dummies(df["Education"], prefix="Education")
interest_encoded = pd.get_dummies(df["Interest"], prefix="Interest")

# Combine all features
X = pd.concat([education_encoded, interest_encoded, skills_encoded], axis=1)
y = df["Internship"]  # Target variable

# Train-test split (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save artifacts
joblib.dump(model, "model.pkl")
joblib.dump(mlb, "skills_encoder.pkl")
joblib.dump(X.columns, "model_columns.pkl")

print("✅ Model trained and saved!")
print(f"Model Accuracy: {model.score(X_test, y_test):.2f}")