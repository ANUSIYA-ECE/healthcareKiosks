import re
import pandas as pd

SYMPTOMS = pd.read_csv("data/Symptom_severity.csv")["Symptom"].str.lower().tolist()

def extract_symptoms(text):
    text = text.lower()
    found = []
    for symptom in SYMPTOMS:
        if symptom.replace("_", " ") in text:
            found.append(symptom.replace(" ", "_"))
    return list(set(found))

def extract_duration(text):
    # Extract number + time unit (days, weeks)
    text = text.lower()
    match = re.search(r"(\d+)\s*(day|days|week|weeks)", text)
    if match:
        num = int(match.group(1))
        unit = match.group(2)
        if "week" in unit:
            return num * 7
        return num
    return 1  # Default to 1 day if unspecified
