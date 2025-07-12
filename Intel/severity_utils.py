import pandas as pd

severity_data = pd.read_csv("data/Symptom_severity.csv")
severity_dict = dict(zip(severity_data["Symptom"].str.lower(), severity_data["weight"]))

def calculate_severity(symptoms, days):
    base_score = sum(severity_dict.get(s.lower(), 0) for s in symptoms)
    if days >= 7:
        adjusted = base_score * 1.5
    elif days >= 3:
        adjusted = base_score * 1.2
    else:
        adjusted = base_score
    avg = adjusted / len(symptoms) if symptoms else 0
    return round(adjusted), round(avg, 1)

def classify_urgency(avg_score):
    if avg_score >= 6:
        return "⚠️ High severity. Urgent consultation recommended."
    elif avg_score >= 4:
        return "⚠️ Moderate severity. You should consult a doctor soon."
    elif avg_score >= 2:
        return "🩺 Mild severity. Monitor your symptoms."
    else:
        return "🙂 Very mild symptoms. Self-care may be enough."
