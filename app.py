import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Disease Predictor",
    page_icon="⚕️",
    layout="centered"
)

# --- 2. THE AI'S MEDICAL DATA ---
@st.cache_data
def load_and_prep_data():
    symptoms = [
        "itching", "skin_rash", "continuous_sneezing", "shivering", "chills", "joint_pain",
        "stomach_pain", "acidity", "vomiting", "fatigue", "weight_gain", "anxiety",
        "cold_hands_and_feets", "mood_swings", "weight_loss", "lethargy", "cough",
        "high_fever", "breathlessness", "sweating", "dehydration", "indigestion", "headache"
    ]
    
    data = {
        "Fungal Infection":     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Allergy":              [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "GERD (Acidity)":       [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "Chronic Cholestasis":  [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Drug Reaction":        [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Peptic Ulcer Disease": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "Migraine":             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "Malaria":              [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        "Chickenpox":           [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        "Dengue":               [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "Common Cold":          [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
    }
    
    rows, labels = [], []
    for disease, matrix in data.items():
        for _ in range(50): 
            noise = np.random.choice([0, 1], size=len(matrix), p=[0.92, 0.08]) 
            mutated_matrix = np.clip(np.array(matrix) + noise, 0, 1)
            rows.append(mutated_matrix)
            labels.append(disease)
            
    df = pd.DataFrame(rows, columns=symptoms)
    df['prognosis'] = labels
    return df, symptoms

df, symptom_list = load_and_prep_data()

# --- 3. TRAINING THE AI MODEL ---
X = df[symptom_list]
y = df['prognosis']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

# --- 4. THE WEB PAGE INTERFACE ---
st.title("⚕️ AI-Powered Disease Detection Assistant")
st.write("Select your active symptoms below to check potential conditions.")
st.info("⚠️ **Disclaimer:** For educational use only. Does not replace clinical consultation.")

st.subheader("📋 Select Symptoms")
selected_symptoms = st.multiselect("Pick symptoms from the menu:", options=symptom_list)

if st.button("Analyze Symptoms", type="primary"):
    if not selected_symptoms:
        st.warning("Please pick at least one symptom.")
    else:
        input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
        input_df = pd.DataFrame([input_data], columns=symptom_list)
        
        probabilities = model.predict_proba(input_df)[0]
        top_indices = np.argsort(probabilities)[::-1][:3]
        
        st.success("### 📊 Diagnostic Evaluation Results")
        for idx in top_indices:
            confidence = probabilities[idx] * 100
            disease_name = le.inverse_transform([idx])[0]
            if confidence > 0:
                st.write(f"**{disease_name}**")
                st.progress(int(confidence))
                st.write(f"Match Likelihood: `{confidence:.2f}%`")
                st.markdown("---")
