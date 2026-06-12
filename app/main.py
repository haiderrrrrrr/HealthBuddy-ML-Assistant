from flask import Flask, request, jsonify
from flask import send_from_directory
import torch
import numpy as np
import os, datetime
from fpdf import FPDF
import time
from flask import render_template
try:
    from .symptom_model_loader import load_symptom_model
    from .lifestyle_model_loader import load_lifestyle_model
    from .medquad_retriever import retrieve_medquad
    from .symptom_text_to_vector import symptoms_to_vector, extract_symptoms
    from .condition_labels import load_condition_labels
except ImportError:
    from symptom_model_loader import load_symptom_model
    from lifestyle_model_loader import load_lifestyle_model
    from medquad_retriever import retrieve_medquad
    from symptom_text_to_vector import symptoms_to_vector, extract_symptoms
    from condition_labels import load_condition_labels
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

symptom_model = load_symptom_model()
lifestyle_model = load_lifestyle_model()
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
COND_LABELS = load_condition_labels(BASE_DIR)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict/symptom-risk", methods=["POST"])
def symptom_predict():
    data = request.get_json(silent=True) or {}
    if "symptom_vector" not in data:
        return jsonify({"error": "symptom_vector is required"}), 400
    vec = np.array(data["symptom_vector"], dtype=np.float32)
    expected = symptom_model.net[0].in_features
    if vec.ndim != 1 or vec.size != expected:
        return jsonify({"error": f"symptom_vector must contain {expected} values"}), 400

    with torch.no_grad():
        out = symptom_model(torch.tensor(vec).unsqueeze(0))
    return jsonify({"risk_vector": out.squeeze().tolist()})


@app.route("/predict/lifestyle-risk", methods=["POST"])
def lifestyle_predict():
    data = request.get_json(silent=True) or {}
    if "lifestyle_features" not in data:
        return jsonify({"error": "lifestyle_features is required"}), 400
    vec_in = np.array(data["lifestyle_features"], dtype=np.float32)
    if vec_in.ndim != 1 or vec_in.size == 0:
        return jsonify({"error": "lifestyle_features must be a non-empty list"}), 400
    in_features = getattr(lifestyle_model.layers[0], "in_features", vec_in.size)
    if vec_in.size >= 6:
        b0, a1 = float(vec_in[0]), float(vec_in[1])
        looks_bmi_age = (10.0 <= b0 <= 60.0) and (0.0 <= a1 <= 120.0)
        if looks_bmi_age:
            ordered = [a1, b0, float(vec_in[2]), float(vec_in[3]), float(vec_in[4]), float(vec_in[5])]
            vec_in = np.array(ordered, dtype=np.float32)
    if vec_in.size < in_features:
        pad = np.zeros(in_features, dtype=np.float32)
        pad[:vec_in.size] = vec_in
        vec = pad
    else:
        vec = vec_in[:in_features]

    with torch.no_grad():
        logits = lifestyle_model(torch.tensor(vec).unsqueeze(0))
        T = 2.5
        probs = torch.softmax(logits / T, dim=1).squeeze(0).tolist()

    bmi = float(vec[1]) if vec.size > 1 else 0.0
    age = float(vec[0]) if vec.size > 0 else 0.0
    sleep = float(vec[2]) if vec.size > 2 else 0.0
    smoking = float(vec[3]) if vec.size > 3 else 0.0
    activity = float(vec[4]) if vec.size > 4 else 0.0
    bp = float(vec[5]) if vec.size > 5 else 0.0

    probs = _calibrate_lifestyle_probs(vec, probs)

    s = sum(probs)
    if s > 0:
        probs = [p / s for p in probs]
    pred_class = int(np.argmax(probs))
    labels = _labels_for_lifestyle()
    risk_label = labels[pred_class]
    summary_sentence = (
        f"Age {int(age)}, BMI {round(float(bmi),1)}, sleep {round(float(sleep),1)}h, "
        f"{'smoker' if smoking>=0.5 else 'non-smoker'}, {'active' if activity>=0.5 else 'inactive'}, "
        f"systolic BP {int(bp)} mmHg → {risk_label}."
    )
    doctor_lifestyle = _maybe_lifestyle_note_via_gemini(age, bmi, sleep, smoking, activity, bp, risk_label, probs)
    if not doctor_lifestyle:
        doctor_lifestyle = _fallback_lifestyle_plan(age, bmi, sleep, smoking, activity, bp, risk_label)

    return jsonify({
        "risk_class": int(pred_class),
        "class_probs": probs,
        "summary_sentence": summary_sentence,
        "doctor_lifestyle": doctor_lifestyle
    })


@app.route("/retrieve/medquad", methods=["POST"])
def medquad_endpoint():
    data = request.get_json(silent=True) or {}
    q = str(data.get("query", "")).strip()
    if not q:
        return jsonify({"error": "query is required"}), 400
    if len(q) > 1000:
        return jsonify({"error": "query is too long"}), 400
    results = retrieve_medquad(q, k=3)
    return jsonify(results)

@app.route("/admin/labels", methods=["GET", "POST"])
def labels_admin():
    configured_key = os.getenv("ADMIN_API_KEY")
    supplied_key = request.headers.get("X-Admin-Key")
    if not configured_key or supplied_key != configured_key:
        return jsonify({"error": "Admin authorization required"}), 403
    base = os.path.join(BASE_DIR, "datasets", "symcat_400")
    os.makedirs(base, exist_ok=True)
    fp = os.path.join(base, "labels.json")
    if request.method == "POST":
        try:
            data = request.get_json(force=True)
            if not isinstance(data, dict):
                return jsonify({"error": "JSON body must be an object of index→name"}), 400
            with open(fp, "w", encoding="utf-8") as f:
                import json
                json.dump({str(k): v for k, v in data.items()}, f, ensure_ascii=False, indent=2)
            global COND_LABELS
            COND_LABELS = load_condition_labels(BASE_DIR)
            return jsonify({"status": "ok", "count": len(COND_LABELS)})
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    else:
        return jsonify({
            "exists": os.path.exists(fp),
            "count": len(COND_LABELS)
        })


def _labels_for_lifestyle():
    return ["No Diabetes", "Prediabetes", "Diabetes"]

def _urdu_block(pred_class_label):
    return (
        "Yeh report maloomati maqsad ke liye hai, tashkhees nahin.\n"
        f"Mutawaqqa khatra: {pred_class_label}.\n"
        "Mashwara: rozana halki warzish, munasib neend, sehatmand ghiza aur pani ka istemal barhayein."
    )

def _calibrate_lifestyle_probs(vec, probs):
    age = float(vec[0]) if len(vec) > 0 else 0.0
    bmi = float(vec[1]) if len(vec) > 1 else 0.0
    sleep = float(vec[2]) if len(vec) > 2 else 0.0
    smoking = float(vec[3]) if len(vec) > 3 else 0.0
    activity = float(vec[4]) if len(vec) > 4 else 0.0
    bp = float(vec[5]) if len(vec) > 5 else 0.0
    p0, p1, p2 = probs
    f_p = 1.0
    if bmi < 27: f_p *= 0.70
    if sleep >= 7: f_p *= 0.85
    if smoking <= 0.1: f_p *= 0.90
    if activity >= 0.5: f_p *= 0.85
    if bp < 130: f_p *= 0.85
    if age < 45: f_p *= 0.90
    f_r = 1.0
    if bmi >= 30: f_r *= 1.20
    if sleep < 6: f_r *= 1.10
    if smoking >= 0.5: f_r *= 1.10
    if activity < 0.5: f_r *= 1.10
    if bp >= 140: f_r *= 1.15
    if age >= 55: f_r *= 1.10
    f = f_p * f_r
    p2 = min(p2 * f, 0.80)
    if f < 1.0:
        p0 *= 1.50
        p1 *= 1.20
    s = p0 + p1 + p2
    if s > 0:
        p0 /= s
        p1 /= s
        p2 /= s
    return [p0, p1, p2]

def _maybe_lifestyle_note_via_gemini(age, bmi, sleep, smoking, activity, bp, risk_label, probs):
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        model_name_order = ["gemini-2.5-flash", "gemini-1.5-flash"]
        smoking_txt = "smoker" if smoking >= 0.5 else "non-smoker"
        activity_txt = "active" if activity >= 0.5 else "inactive"
        prompt = (
            "You are a physician coach. Write a concise, positive lifestyle plan for the patient.\n"
            f"Metrics: age {int(age)}, BMI {round(float(bmi),1)}, sleep {round(float(sleep),1)}h, {smoking_txt}, {activity_txt}, systolic BP {int(bp)} mmHg.\n"
            f"Current risk: {risk_label} (probs: No {probs[0]:.2f}, Pre {probs[1]:.2f}, Dia {probs[2]:.2f}).\n"
            "Deliver: a brief routine with exercise, diet, sleep, hydration, and stress guidance; non-prescriptive, safe, and motivating. End with a gentle follow-up note."
        )
        for attempt in range(4):
            for mn in model_name_order:
                try:
                    model = genai.GenerativeModel(mn)
                    resp = model.generate_content(prompt)
                    txt = (getattr(resp, 'text', '') or '').strip() or None
                    if txt:
                        return txt
                except Exception:
                    time.sleep(0.5 + attempt * 0.5)
                    continue
        return None
    except Exception:
        return None

def _fallback_lifestyle_plan(age, bmi, sleep, smoking, activity, bp, risk_label):
    lines = []
    lines.append("Daily Plan: 30–45 minutes of brisk walking or light cardio.")
    if bmi >= 25:
        lines.append("Diet: emphasize vegetables, lean protein, whole grains; limit sugar and refined carbs.")
    else:
        lines.append("Diet: balanced meals with whole foods; maintain portion control.")
    if sleep < 7:
        lines.append("Sleep: aim for 7–8 hours with a consistent bedtime and reduced screen time.")
    if smoking >= 0.5:
        lines.append("Smoking: begin a quit plan; use support resources and avoid triggers.")
    if activity < 0.5:
        lines.append("Activity: add two strength sessions weekly for metabolic health.")
    if bp >= 130:
        lines.append("BP: reduce salt, manage stress, and monitor blood pressure weekly.")
    lines.append("Hydration: 6–8 glasses of water daily; limit sugary drinks.")
    lines.append("Follow-up: review progress in 2–4 weeks; seek clinical advice if concerns persist.")
    return " \n".join(lines)

def _maybe_doctor_note_via_gemini(text, matched, top_names, top_scores, medquad_results):
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        model_name_order = ["gemini-2.5-flash", "gemini-1.5-flash"]
        med_lines = []
        for r in (medquad_results or [])[:3]:
            q = r.get('question','')
            a = (r.get('answer','') or '')[:220]
            med_lines.append(f"- {q}: {a}")
        prompt = (
            "You are a careful physician writing to a patient.\n"
            f"Patient symptom text: {text}\n"
            f"Matched symptoms: {', '.join(matched) if matched else 'none'}\n"
            f"Top conditions: {', '.join(top_names) if top_names else 'none'}\n"
            f"Evidence (MedQuAD):\n" + ("\n".join(med_lines) if med_lines else "- none") + "\n\n"
            "Write a detailed, empathetic guidance that:\n"
            "- Starts with a one-sentence definition of the top condition (what it is), without percentages\n"
            "- Explains why the top condition is most likely based on symptoms\n"
            "- Describes how symptoms connect clinically to the top condition\n"
            "- Explains why other conditions are less likely given the symptom profile\n"
            "- Provides stepwise care advice and typical medicine guidance (non-prescriptive)\n"
            "- Encourages follow-up and highlights gentle watchouts, avoiding alarmist tone\n"
        )
        for attempt in range(5):
            for model_name in model_name_order:
                try:
                    model = genai.GenerativeModel(model_name)
                    resp = model.generate_content(prompt)
                    txt = (getattr(resp, 'text', '') or '').strip() or None
                    if txt:
                        return txt
                except Exception as e:
                    if "503" in str(e):
                        time.sleep(1 + attempt)
                        continue
                    continue
        return None
    except Exception:
        return None

@app.route("/healthbuddy/report", methods=["POST"])
def report():
    data = request.json or {}
    preview = bool(data.get("preview", False))
    sym_vec = np.array(data.get("symptom_vector", []), dtype=np.float32)
    life_vec = np.array(data.get("lifestyle_features", []), dtype=np.float32)
    text = str(data.get("symptom_text", "")).lower()
    if text and sym_vec.size == 0:
        sym_vec = symptoms_to_vector(text)
    matched_local = extract_symptoms(text)
    sym_pos = bool(sym_vec.size > 0 and np.any(sym_vec > 0))
    if life_vec.size == 0:
        in_features = getattr(lifestyle_model.layers[0], "in_features", 21)
        lv = np.zeros(in_features, dtype=np.float32)
        if "age" in data:
            lv[0] = float(data.get("age", 0))
        if "bmi" in data:
            lv[1] = float(data.get("bmi", 0))
        if "sleep_hours" in data:
            lv[2] = float(data.get("sleep_hours", 0))
        if "smoking" in data:
            lv[3] = float(data.get("smoking", 0))
        if "phys_activity" in data:
            lv[4] = float(data.get("phys_activity", 0))
        if "systolic_bp" in data:
            lv[5] = float(data.get("systolic_bp", 0))
        life_vec = lv

    eng_sections = []
    alert_message = ""
    top_names = []
    top_scores = []
    top = []
    lead = None
    lead_score = None

    if sym_pos:
        with torch.no_grad():
            rv = symptom_model(torch.tensor(sym_vec).unsqueeze(0))
            prob_vec = torch.softmax(rv, dim=1).squeeze(0).tolist()
        top = np.argsort(prob_vec)[-5:][::-1]
        top_names = [COND_LABELS.get(int(i), f"Condition {int(i)}") for i in top]
        top_scores = [prob_vec[int(i)] for i in top]
        desc = []
        if "cough" in text:
            desc.append("Respiratory stress suggested; increase hydration and avoid smoke exposure.")
        if "fever" in text:
            desc.append("Fever noted; consider rest and monitor temperature.")
        if "headache" in text:
            desc.append("Headache reported; manage stress and hydration.")
        if "fatigue" in text:
            desc.append("Fatigue present; improve sleep hygiene and nutrition.")
        if "sore throat" in text:
            desc.append("Sore throat; warm fluids and rest.")
        if "shortness of breath" in text:
            desc.append("Shortness of breath; consider breathing exercises and avoid exertion.")
        lead = top_names[0] if top_names else None
        lead_score = top_scores[0] if top_scores else None
        if lead:
            eng_sections.append(
                f"Clinical Summary: The most likely condition is {lead}. "
                + (" ".join(desc) if desc else "General check recommended.")
            )
            if len(top_names) > 1:
                others = ", ".join([f"{n}" for i, n in enumerate(top_names[1:])])
                eng_sections.append(f"Other possibilities: {others}.")
        else:
            eng_sections.append("Clinical Summary: No clear condition detected; provide more specific symptoms.")

        top_def = None
        try:
            if lead and not preview:
                defs = retrieve_medquad(f"What is {lead}", k=1)
                if defs:
                    a = (defs[0].get('answer','') or '').strip()
                    if a:
                        top_def = a[:300]
        except Exception:
            pass
        if lead:
            others_list = [
                f"{top_names[i]}" for i in range(1, min(3, len(top_names)))
            ]
            others_txt = (", ".join(others_list)) if others_list else "none"
            doctor_note = (
                f"Doctor: Based on your reported symptoms ({', '.join(matched_local) or 'none'}), "
                f"the most likely condition is {lead}. "
                f"What it is: {top_def or (lead + ' is a clinical condition; a clinician can explain specifics based on your presentation.') } "
                f"This ranking reflects strong alignment between your symptoms and its common clinical pattern. "
                f"Other possibilities include {others_txt}, but they are less likely given the symptom profile. "
                f"Care steps: 1) Rest and adequate fluids; 2) Monitor temperature and key symptoms; 3) Avoid triggers (smoke, allergens); 4) Seek clinical evaluation if worsening or red flags occur. "
                f"Medicine guidance: Consider appropriate OTC relief when suitable; a clinician can advise prescriptions if indicated."
            )
        else:
            doctor_note = "Based on the information provided, we couldn't match any symptom with sufficient confidence. No condition is suggested. Please enter more specific symptom phrases."
    else:
        alert_message = "Based on the information provided, we couldn't match any symptom with sufficient confidence. No condition is suggested. Please enter more specific symptom phrases."
        if text.strip():
            eng_sections.append(alert_message)
        doctor_note = alert_message

    pred_label = None
    summary_sentence = None
    doctor_life_note = None
    if life_vec.size > 0:
        with torch.no_grad():
            logits = lifestyle_model(torch.tensor(life_vec).unsqueeze(0))
            T = 2.5
            probs = torch.softmax(logits / T, dim=1).squeeze(0).tolist()
        bmi = float(life_vec[1]) if life_vec.size > 1 else 0.0
        age = float(life_vec[0]) if life_vec.size > 0 else 0.0
        sleep = float(life_vec[2]) if life_vec.size > 2 else 0.0
        smoking = float(life_vec[3]) if life_vec.size > 3 else 0.0
        activity = float(life_vec[4]) if life_vec.size > 4 else 0.0
        bp = float(life_vec[5]) if life_vec.size > 5 else 0.0
        probs = _calibrate_lifestyle_probs(life_vec, probs)
        s = sum(probs)
        if s > 0:
            probs = [p / s for p in probs]
        pred = int(np.argmax(probs))
        labels = _labels_for_lifestyle()
        pred_label = labels[pred]
        summary_sentence = (
            f"Age {int(age)}, BMI {round(float(bmi),1)}, sleep {round(float(sleep),1)}h, "
            f"{'smoker' if smoking>=0.5 else 'non-smoker'}, {'active' if activity>=0.5 else 'inactive'}, "
            f"systolic BP {int(bp)} mmHg → {pred_label}."
        )
        try:
            doctor_life_note = _maybe_lifestyle_note_via_gemini(age, bmi, sleep, smoking, activity, bp, pred_label, probs)
        except Exception:
            doctor_life_note = None
        if not doctor_life_note:
            doctor_life_note = _fallback_lifestyle_plan(age, bmi, sleep, smoking, activity, bp, pred_label)

    matched = matched_local
    query = data.get("query") or (" ".join(matched) + " symptoms" if matched else "general health advice")
    results = []
    if not preview:
        results = retrieve_medquad(query, k=3)
        if results:
            eng_sections.append("MedQuAD Guidance:")
            for r in results:
                eng_sections.append(
                    f"- {r.get('question','')}\n  {r.get('answer','')[:220]}..."
                )

    report_lines = []
    report_lines.append("HealthBuddy Report")
    report_lines.append("")
    report_lines.append("Step 1 — Symptoms")
    report_lines.append(f"Matched Symptoms: {', '.join(matched) if matched else 'none'}")
    if lead:
        report_lines.append("Top Conditions:")
        for i, n in enumerate(top_names):
            sc = top_scores[i] if i < len(top_scores) else None
            sc_txt = f" ({sc:.2f})" if sc is not None else ""
            report_lines.append(f"- {n}{sc_txt}")
    if doctor_note:
        report_lines.append("Doctor Note:")
        report_lines.append(doctor_note)
    report_lines.append("")
    report_lines.append("Step 2 — Lifestyle")
    if pred_label:
        report_lines.append(f"Risk: {pred_label}")
    if summary_sentence:
        report_lines.append(summary_sentence)
    if doctor_life_note:
        report_lines.append("Lifestyle Advice:")
        report_lines.append(doctor_life_note)
    report_lines.append("")
    report_lines.append("Step 3 — Medical Query")
    if query:
        report_lines.append(f"Your Question: {query}")
    if results:
        report_lines.append("MedQuAD Guidance:")
        for r in results:
            q = r.get('question','')
            a = (r.get('answer','') or '')[:220]
            src = r.get('source','') or ''
            line = f"- {q}\n  {a}..."
            if src:
                line += f"\n  Source: {src}"
            report_lines.append(line)
    report_lines.append("")
    report_lines.append("Disclaimer: This is risk screening only and not medical diagnosis.")
    english_text = "\n".join(report_lines)
    urdu_text = _urdu_block(pred_label or "General Lifestyle Risk")

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "stored_models", "Reports", f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "health_report.pdf")

    download_url = None
    if not preview:
        pdf = FPDF()
        pdf.add_page()
        try:
            # Use lowercase 'fonts' directory to avoid cross-platform issues
            pdf.add_font("DejaVu", "", os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts", "DejaVuSans.ttf"))
            pdf.set_font("DejaVu", size=12)
        except Exception:
            pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, english_text + "\n\n" + urdu_text)
        pdf.output(pdf_path)
        # Provide a public download URL
        run_dir = f"run_{ts}"
        download_url = f"/download/report/{run_dir}/health_report.pdf"

    # Try Gemini note if available
    try:
        gem_note = None
        if not preview:
            gem_note = _maybe_doctor_note_via_gemini(text, matched_local, top_names if sym_pos else [], top_scores if sym_pos else [], results if 'results' in locals() else [])
        if gem_note:
            doctor_note = gem_note
    except Exception:
        pass

    return jsonify({
        "english": english_text,
        "urdu": urdu_text,
        "pdf_path": pdf_path if not preview else None,
        "download_url": download_url,
        "matched_symptoms": matched_local,
        "top_conditions": top_names if sym_pos else [],
        "top_condition_ids": [int(i) for i in top] if sym_pos else [],
        "top_scores": top_scores if sym_pos else [],
        "doctor_note": doctor_note,
        "labels_present": len(COND_LABELS) > 0,
        "no_symptom_match": not sym_pos,
        "alert_message": alert_message
    })

@app.route("/download/report/<run_dir>/<filename>", methods=["GET"])
def download_report(run_dir, filename):
    base_reports = os.path.join(os.path.dirname(os.path.dirname(__file__)), "stored_models", "Reports")
    if not run_dir.startswith("run_") or filename != "health_report.pdf":
        return jsonify({"error": "Invalid report path"}), 400
    safe_dir = os.path.join(base_reports, run_dir)
    if not os.path.isfile(os.path.join(safe_dir, filename)):
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(safe_dir, filename, as_attachment=True, download_name=filename)

@app.route("/")
def home():
    # Front page: welcome screen with background GIF
    return render_template("welcome.html")

@app.route("/index")
def index_page():
    # Main Health Buddy interactive page
    return render_template("index.html")

@app.route("/dashboard")
def dashboard_page():
    return render_template("userDashboard.html")

@app.route("/dashboard/chart-data", methods=["GET"])
def dashboard_chart_data():
    # Provide HealthBuddy system chart data for the dashboard
    data = {
        "history_dates": ["2025-01-01","2025-02-15","2025-03-20","2025-04-10","2025-05-01"],
        "history_scores": [2.5, 3.0, 2.8, 3.2, 3.1],
        "confidence_labels": ["Scan 1","Scan 2","Scan 3","Scan 4","Scan 5"],
        "confidence_values": [70, 85, 60, 90, 75],
        "radar_labels": ["Upper Left","Upper Right","Lower Left","Lower Right","Central"],
        "radar_values": [3.0, 2.8, 3.2, 2.9, 3.1],
        "benchmark_labels": ["You","Population Avg"],
        "benchmark_values": [3.2, 2.5],
        "result_labels": ["Benign","Malignant","Inconclusive"],
        "result_counts": [60, 30, 10],
        "time_bins": ["0–30 days","31–60 days","61–90 days","91–120 days"],
        "time_counts": [2, 3, 1, 1]
    }
    return jsonify(data)

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/faq")
def faq_page():
    return render_template("faq.html")

@app.route("/privacy")
def privacy_page():
    return render_template("privacy.html")

@app.route("/predictor")
def predictor_page():
    return render_template("predictor.html")

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
