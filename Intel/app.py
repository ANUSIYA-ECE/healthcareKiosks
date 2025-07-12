from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from difflib import get_close_matches
import torch
import re

app = Flask(__name__)

# Load model & encoders
with open("model.pkl", "rb") as f:
    model, mlb, y_cat = pickle.load(f)

# Load additional data
desc = pd.read_csv("data/symptom_Description.csv")
prec = pd.read_csv("data/symptom_precaution.csv")
severity_df = pd.read_csv("data/Symptom_severity.csv")

# Sentence Transformer model
embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Symptom data
known_symptoms = list(mlb.classes_)
symptom_embeddings = embedder.encode(known_symptoms, convert_to_tensor=True)

# Helper Functions
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def extract_duration(text):
    match = re.search(r"(\d+)\s*(day|days|week|weeks)", text)
    if match:
        num = int(match.group(1))
        unit = match.group(2)
        return num * 7 if "week" in unit else num
    return None

def semantic_symptom_match(text):
    input_embedding = embedder.encode([text], convert_to_tensor=True)[0]
    cos_scores = util.cos_sim(input_embedding, symptom_embeddings)[0]
    top_results = torch.topk(cos_scores, k=5)
    matched = []
    for score, idx in zip(top_results.values, top_results.indices):
        if score.item() > 0.6:
            matched.append(known_symptoms[idx])
    return matched

def fuzzy_symptom_match(symptom):
    matches = get_close_matches(symptom, known_symptoms, n=1, cutoff=0.8)
    return matches[0] if matches else None

def get_description(disease):
    row = desc[desc["Disease"].str.lower() == disease.lower()]
    return row["Description"].values[0] if not row.empty else "No description found."

def get_precautions(disease):
    row = prec[prec["Disease"].str.lower() == disease.lower()]
    if not row.empty:
        return [row[f"Precaution_{i}"].values[0] for i in range(1, 5) if pd.notna(row[f"Precaution_{i}"].values[0])]
    return ["No precautions found."]

def get_severity_score(symptoms):
    sev_map = dict(zip(severity_df["Symptom"], severity_df["weight"]))
    scores = [sev_map.get(s, 3) for s in symptoms]
    return sum(scores), round(sum(scores)/len(scores), 1) if scores else 0

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = request.form["symptoms"]
        cleaned = clean_text(user_input)

        duration = extract_duration(cleaned)
        words = cleaned.split()

        matched_symptoms = set()

        for phrase in re.split(r"[,.]", cleaned):
            matched_symptoms.update(semantic_symptom_match(phrase.strip()))

        # Also try fuzzy matching on each word
        for word in words:
            fuzzy = fuzzy_symptom_match(word)
            if fuzzy:
                matched_symptoms.add(fuzzy)

        matched_symptoms = list(matched_symptoms)

        if not matched_symptoms:
            return jsonify({"reply": "⚠️ No recognizable symptoms found. Try rephrasing your input."})
        if len(matched_symptoms) == 1:
            return jsonify({"reply": f"🩺 Only found: {matched_symptoms[0]}. Please mention more symptoms or duration."})

        input_vector = mlb.transform([matched_symptoms])
        pred_code = model.predict(input_vector)[0]
        disease = y_cat.categories[pred_code]

        description = get_description(disease)
        precautions = get_precautions(disease)

        severity_score, avg_sev = get_severity_score(matched_symptoms)

        if severity_score > 20:
            severity_label = "🚨 Severe symptoms. Please consult a doctor."
        elif severity_score > 10:
            severity_label = "⚠️ Moderate symptoms. Monitor and take precautions."
        else:
            severity_label = "🩺 Mild symptoms. Rest and monitor."

        reply = (
            f"<b>Disease Prediction:</b> {disease}<br><br>"
            f"<b>Description:</b><br>{description}<br><br>"
            f"<b>Precautions:</b><br>" + "<br>".join(precautions) + "<br><br>"
            f"<b>Symptom Count:</b> {len(matched_symptoms)}<br>"
            f"<b>Severity Score:</b> {severity_score} (avg {avg_sev})<br>"
        )

        if duration:
            reply += f"<b>Symptom Duration:</b> {duration} days<br>"

        reply += f"<br>{severity_label}"

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"❌ Error occurred: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
