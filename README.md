# HealthBuddy ML Assistant

HealthBuddy is a Flask-based health screening assistant that combines trained PyTorch models, natural-language symptom matching, lifestyle risk analysis, MedQuAD retrieval, PDF reporting, and optional Gemini-generated guidance.

## Features

- Free-text symptom extraction across common clinical categories
- 1,300-feature symptom vectorization and 400-condition risk scoring
- Lifestyle-based diabetes risk classification
- MedQuAD question-and-answer retrieval with semantic and TF-IDF fallbacks
- Optional Gemini guidance when an API key is configured
- Combined symptom, lifestyle, and medical-query reports
- Downloadable PDF reports
- Browser-based predictor, dashboard, FAQ, privacy, and project information pages
- Model training scripts and stored evaluation artifacts

## How It Works

1. Symptom text is normalized and matched against a curated symptom vocabulary.
2. The symptom model produces scores across 400 condition labels.
3. Lifestyle features are evaluated by a separate three-class PyTorch model.
4. Medical questions retrieve relevant MedQuAD entries.
5. HealthBuddy combines the results into an informational report and optional PDF.

HealthBuddy is a screening and educational project. It does not provide a medical diagnosis or replace professional medical care.

## Technology Stack

| Area | Technologies |
| --- | --- |
| Web application | Python, Flask, HTML, CSS, JavaScript |
| Machine learning | PyTorch, NumPy, scikit-learn |
| Retrieval | Sentence Transformers, TF-IDF, MedQuAD |
| Generative guidance | Google Gemini API (optional) |
| Reporting | FPDF2 |

## Project Structure

```text
healthbuddy-ml-assistant/
|-- app/
|   |-- static/              # Styles, scripts, and interface assets
|   |-- templates/           # Flask pages
|   |-- main.py              # Routes and report workflow
|   |-- medquad_retriever.py
|   |-- symptom_model_loader.py
|   `-- symptom_text_to_vector.py
|-- datasets/                # SymCAT, BRFSS, and MedQuAD data
|-- models/                  # PyTorch model definitions
|-- stored_models/           # Trained weights, indexes, and metrics
|-- tests/                   # API and model checks
|-- training/                # Model and retrieval-index training scripts
|-- requirements.txt         # Web runtime dependencies
|-- requirements-training.txt
`-- run.py                   # Application entry point
```

## Prerequisites

- Python 3.10-3.12
- A virtual environment
- At least 2 GB of free disk space for dependencies and model assets

## Local Setup

Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env`. Gemini is optional:

```env
GEMINI_API_KEY=
ADMIN_API_KEY=replace_with_a_long_random_value
FLASK_DEBUG=0
PORT=8000
```

Start the application:

```bash
python run.py
```

Open `http://localhost:8000`.

## Example Inputs

Symptom description:

```text
I have fever, cough, headache, fatigue, and a sore throat.
```

Lifestyle profile:

```text
Age: 42
BMI: 27.5
Sleep: 6.5 hours
Smoking: No
Physical activity: Yes
Systolic blood pressure: 128
```

Medical query:

```text
What are the common symptoms and risk factors for diabetes?
```

## API Overview

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Service health check |
| `/predict/symptom-risk` | POST | Score a 1,300-value symptom vector |
| `/predict/lifestyle-risk` | POST | Classify lifestyle risk |
| `/retrieve/medquad` | POST | Retrieve relevant medical Q&A |
| `/healthbuddy/report` | POST | Build a combined report |
| `/download/report/...` | GET | Download a generated report |

The `/admin/labels` endpoint requires `ADMIN_API_KEY` in the `X-Admin-Key` request header.

## Tests

```bash
python -m unittest tests.test_api -v
python tests/test.py
```

## Model Training

Install the extended dependencies:

```bash
pip install -r requirements-training.txt
```

Run the training entry points:

```bash
python training/train_symptom_model.py
python training/train_lifestyle_model.py
python training/retriever_medquad.py
```

Training outputs are written beneath `stored_models/`.

## Deployment

The application requires a persistent Python service and sizeable model files. A container or Render Web Service is more suitable than a serverless function.

Production command:

```bash
gunicorn run:app
```

Set `GEMINI_API_KEY` only when generative guidance is required. Deterministic fallback guidance remains available without it.
