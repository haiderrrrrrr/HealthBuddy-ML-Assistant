"""
API TESTS — HealthBuddy
Validates Flask endpoints and end-to-end report generation.

Run:
  python tests/test_api.py
"""

import os
import json
import unittest
import numpy as np

# Ensure project root on path for imports
import sys
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
APP_DIR = os.path.join(BASE_DIR, "app")
for p in (BASE_DIR, APP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Preload root-level modules that app.main imports using relative names
import importlib
importlib.import_module('symptom_model_loader')
importlib.import_module('lifestyle_model_loader')
importlib.import_module('condition_labels')

from app.main import app, lifestyle_model, symptom_model  # type: ignore


class HealthBuddyApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_symptom_risk_endpoint(self):
        vec = np.zeros(1300, dtype=np.float32)
        vec[101] = 1  # fever
        vec[502] = 1  # cough
        resp = self.client.post(
            "/predict/symptom-risk",
            data=json.dumps({"symptom_vector": vec.tolist()}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("risk_vector", data)
        rv = data["risk_vector"]
        self.assertEqual(len(rv), 400)
        # Values should be in [0,1] due to sigmoid
        self.assertTrue(all(0.0 <= x <= 1.0 for x in rv))

    def test_lifestyle_risk_endpoint(self):
        in_features = getattr(lifestyle_model.layers[0], "in_features", 6)
        # Healthy-ish profile
        healthy = np.zeros(in_features, dtype=np.float32)
        if in_features >= 1: healthy[0] = 24   # age
        if in_features >= 2: healthy[1] = 22   # bmi
        if in_features >= 3: healthy[2] = 8    # sleep
        if in_features >= 4: healthy[3] = 0  # non-smoker
        if in_features >= 5: healthy[4] = 1  # active
        if in_features >= 6: healthy[5] = 115

        resp = self.client.post(
            "/predict/lifestyle-risk",
            data=json.dumps({"lifestyle_features": healthy.tolist()}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("risk_class", data)
        self.assertIn("class_probs", data)
        self.assertIn("summary_sentence", data)
        probs = data["class_probs"]
        self.assertEqual(len(probs), 3)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
        self.assertAlmostEqual(sum(probs), 1.0, places=2)
        # Expect No Diabetes highest for healthy profile
        self.assertGreater(probs[0], probs[2])
        self.assertLess(probs[2], 0.65)

    def test_lifestyle_risk_unhealthy_profile(self):
        in_features = getattr(lifestyle_model.layers[0], "in_features", 6)
        unhealthy = np.zeros(in_features, dtype=np.float32)
        if in_features >= 1: unhealthy[0] = 60   # age
        if in_features >= 2: unhealthy[1] = 35   # bmi
        if in_features >= 3: unhealthy[2] = 5    # sleep
        if in_features >= 4: unhealthy[3] = 1    # smoker
        if in_features >= 5: unhealthy[4] = 0    # inactive
        if in_features >= 6: unhealthy[5] = 150  # high bp

        resp = self.client.post(
            "/predict/lifestyle-risk",
            data=json.dumps({"lifestyle_features": unhealthy.tolist()}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        probs = data["class_probs"]
        # Expect Diabetes highest for unhealthy profile
        self.assertGreater(probs[2], probs[0])
        self.assertGreater(probs[2], probs[1])

    def test_lifestyle_risk_borderline_profile(self):
        in_features = getattr(lifestyle_model.layers[0], "in_features", 6)
        borderline = np.zeros(in_features, dtype=np.float32)
        if in_features >= 1: borderline[0] = 46
        if in_features >= 2: borderline[1] = 28
        if in_features >= 3: borderline[2] = 7
        if in_features >= 4: borderline[3] = 0
        if in_features >= 5: borderline[4] = 1
        if in_features >= 6: borderline[5] = 128

        resp = self.client.post(
            "/predict/lifestyle-risk",
            data=json.dumps({"lifestyle_features": borderline.tolist()}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        probs = data["class_probs"]
        self.assertLess(probs[2], 0.60)

    def test_medquad_retrieval_endpoint(self):
        resp = self.client.post(
            "/retrieve/medquad",
            data=json.dumps({"query": "diabetes symptoms"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        results = resp.get_json()
        self.assertIsInstance(results, list)
        # At least one result expected when index is present; fallback may still return >=1
        self.assertGreaterEqual(len(results), 1)
        item = results[0]
        self.assertIn("question", item)
        self.assertIn("answer", item)

    def test_report_generation(self):
        payload = {
            "symptom_text": "fever, cough, headache",
            "age": 29,
            "bmi": 27,
            "sleep_hours": 6,
            "smoking": 0,
            "phys_activity": 1,
            "systolic_bp": 122,
            "query": "diabetes symptoms",
        }
        resp = self.client.post(
            "/healthbuddy/report",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("english", data)
        self.assertIn("urdu", data)
        self.assertIn("pdf_path", data)
        self.assertTrue(data["english"].strip() != "")
        self.assertTrue(data["urdu"].strip() != "")
        # Matched symptoms should include at least fever/cough/headache
        matched = data.get("matched_symptoms", [])
        self.assertTrue(set(["fever", "cough", "headache"]) & set(matched))
        # PDF path should have been created
        pdf_path = data["pdf_path"]
        self.assertTrue(os.path.exists(pdf_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)

