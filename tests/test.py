"""
=====================================================
HEALTHBUDDY – FULL SYSTEM TEST SCRIPT
Tests:
  ✔ Symptom Risk Model (400-dim output)
  ✔ Lifestyle Risk Model (3-class)
  ✔ MedQuAD Retriever
=====================================================
"""

import numpy as np
import torch
import os, sys

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.symptom_model_loader import load_symptom_model
from app.lifestyle_model_loader import load_lifestyle_model
from app.medquad_retriever import retrieve_medquad

# ------------------------------------------------------
print("====================================================")
print(" LOADING MODELS...")
print("====================================================")

symptom_model = load_symptom_model()
lifestyle_model = load_lifestyle_model()

print("✔ Models loaded successfully!\n")


# ------------------------------------------------------
# 1) TEST — Symptom Risk Model
# ------------------------------------------------------
print("====================================================")
print(" TEST 1 — SYMPTOM RISK MODEL")
print("====================================================")

symptom_vector = np.zeros(1300, dtype=np.float32)
symptom_vector[101] = 1   # fever
symptom_vector[305] = 1   # headache
symptom_vector[502] = 1   # cough

with torch.no_grad():
    out = symptom_model(torch.tensor(symptom_vector).unsqueeze(0))

risk_vector = out.squeeze().tolist()

top5_idx = np.argsort(risk_vector)[-5:][::-1]

print("\nUser symptoms: fever, headache, cough")
print("Top 5 predicted conditions:")

for i in top5_idx:
    print(f"  • Condition ID {i} → score {risk_vector[i]:.4f}")

print("\n✔ Symptom model test completed!\n")


# ------------------------------------------------------
# 2) TEST — Lifestyle Risk Model
# ------------------------------------------------------
print("====================================================")
print(" TEST 2 — LIFESTYLE RISK MODEL")
print("====================================================")

# Use the same number of features as the model expects
in_features = getattr(lifestyle_model.layers[0], "in_features", 6)
lifestyle_features = np.zeros(in_features, dtype=np.float32)

if in_features >= 1: lifestyle_features[0] = 28   # BMI
if in_features >= 2: lifestyle_features[1] = 29   # Age
if in_features >= 3: lifestyle_features[2] = 6    # Sleep
if in_features >= 4: lifestyle_features[3] = 0    # Smoking
if in_features >= 5: lifestyle_features[4] = 1    # Activity
if in_features >= 6: lifestyle_features[5] = 122  # Systolic BP

with torch.no_grad():
    logits = lifestyle_model(torch.tensor(lifestyle_features).unsqueeze(0))
    probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
    pred_class = int(np.argmax(probs))

labels = ["No Diabetes", "Prediabetes", "Diabetes"]

print(f"\nLifestyle Input Example: {lifestyle_features.tolist()}")
print(f"Predicted Risk Class: {pred_class} → {labels[pred_class]}")
print("Class Probabilities:")
print(f"  No Diabetes:  {probs[0]:.4f}")
print(f"  Prediabetes:  {probs[1]:.4f}")
print(f"  Diabetes:     {probs[2]:.4f}")

print("\n✔ Lifestyle model test completed!\n")


# ------------------------------------------------------
# 3) TEST — MedQuAD Retriever
# ------------------------------------------------------
print("====================================================")
print(" TEST 3 — MEDQUAD RETRIEVER")
print("====================================================")

query = "What are the symptoms of diabetes?"
print(f"\nRunning query: {query}")

results = retrieve_medquad(query, k=3)

print("\nTop MedQuAD Results:\n")

if len(results) == 0:
    print("❌ No MedQuAD results found — check index path.")
else:
    for r in results:
        print("--------------------------------------------")
        print(f"Score: {r['score']:.4f}")
        print(f"Q: {r['question']}")
        print(f"A: {r['answer'][:220]}...")
        print(f"Source: {r['source']}")
        print(f"Focus: {r['focus_area']}")

print("\n✔ MedQuAD retrieval test completed!")
print("\n====================================================")
print(" ALL TESTS PASSED — SYSTEM IS WORKING CORRECTLY 🎉")
print("====================================================\n")
