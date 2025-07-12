import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/dataset.csv")
df.fillna("", inplace=True)

# Combine all symptom columns into a list
symptom_cols = [col for col in df.columns if col.startswith("Symptom")]
df["Symptoms"] = df[symptom_cols].values.tolist()
df["Symptoms"] = df["Symptoms"].apply(lambda x: [i.strip().lower().replace(" ", "_") for i in x if i != ""])

# Labels
df["Disease"] = df["Disease"].str.strip()
X = df["Symptoms"]
y = df["Disease"].astype("category")
y_cat = y.cat

# Encode symptoms
mlb = MultiLabelBinarizer()
X_encoded = mlb.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X_encoded, y.cat.codes)

# Save model, encoders
with open("model.pkl", "wb") as f:
    pickle.dump((model, mlb, y_cat), f)
  