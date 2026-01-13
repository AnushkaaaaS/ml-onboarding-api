from fastapi import FastAPI, UploadFile, File
import io
from docx import Document
import PyPDF2
import joblib




# ================= ML LOGIC START =================

import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier

# load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- helper functions ----------

def validate_document(text):
    checks = {
        "has_overview": "overview" in text.lower(),
        "has_requirements": "requirement" in text.lower(),
        "has_integrations": "integration" in text.lower(),
        "has_timeline": "week" in text.lower() or "timeline" in text.lower()
    }
    missing = [k for k, v in checks.items() if not v]
    return checks, missing


def chunk_text(text, max_words=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+max_words]))
        i += max_words - overlap
    return chunks


def section_weight(chunk):
    keywords = ["requirement", "deliverable", "integration", "scope", "timeline"]
    return 2.0 if any(k in chunk.lower() for k in keywords) else 1.0


def doc_to_vector(text):
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks)
    weights = np.array([section_weight(c) for c in chunks])
    weights = weights / weights.sum()
    return (embeddings * weights[:, None]).sum(axis=0)


def detect_features(text):
    t = text.lower()
    return {
        "auth": any(k in t for k in ["login", "authentication", "role-based"]),
        "dashboard": "dashboard" in t,
        "notifications": "email" in t or "notification" in t,
        "payments": "payment" in t or "stripe" in t,
        "integrations": "integration" in t or "api" in t
    }


def expand_tasks(category, features):
    tasks = []

    if category == "Backend_Development":
        tasks.append("Design scalable database schema")
        if features["auth"]:
            tasks.append("Implement authentication and role-based access control")
        tasks.append("Develop REST APIs")
        if features["notifications"]:
            tasks.append("Integrate notification services")

    elif category == "Frontend_Development":
        tasks.append("Develop responsive UI")
        if features["dashboard"]:
            tasks.append("Build interactive dashboards")
        if features["auth"]:
            tasks.append("Implement secure login flows")

    elif category == "Payment_Integration" and features["payments"]:
        tasks.append("Configure Stripe sandbox")
        tasks.append("Implement payment workflows")

    elif category == "Testing":
        tasks.append("Write unit tests")
        tasks.append("Perform end-to-end testing")

    elif category == "Deployment":
        tasks.append("Provision cloud infrastructure")
        tasks.append("Deploy to production")

    elif category == "Client Onboarding":
        tasks.append("Send onboarding email")
        tasks.append("Set up project workspace")

    return tasks


# ---- dummy trained model placeholders (REPLACE with your loaded ones) ----
# IMPORTANT: you must load the trained mlb & ovr here if you saved them
mlb = joblib.load("mlb.joblib")
ovr = joblib.load("ovr_model.joblib")


def predict_task_categories(vector, threshold=0.4):
    probs = ovr.predict_proba(vector.reshape(1, -1))[0]
    selected = [mlb.classes_[i] for i, p in enumerate(probs) if p >= threshold]

    if not selected:
        selected = [mlb.classes_[np.argmax(probs)]]

    confidence = {mlb.classes_[i]: float(probs[i]) for i in range(len(probs))}
    return selected, confidence


def clarification_tasks(missing):
    mapping = {
        "has_requirements": "Conduct requirement discovery workshop",
        "has_integrations": "Confirm third-party integrations",
        "has_timeline": "Finalize timeline and milestones"
    }
    return [mapping[m] for m in missing if m in mapping]


def generate_onboarding_plan(document_text):
    _, missing = validate_document(document_text)
    features = detect_features(document_text)

    vector = doc_to_vector(document_text)
    categories, confidence = predict_task_categories(vector)

    plan = []

    clarify = clarification_tasks(missing)
    if clarify:
        plan.append({"phase": "Clarification", "tasks": clarify})

    for cat in categories:
        plan.append({
            "phase": cat,
            "tasks": expand_tasks(cat, features),
            "confidence": round(confidence.get(cat, 0), 2)
        })

    return {
        "plan": plan,
        "predicted_categories": categories,
        "confidence_scores": confidence,
        "missing_sections": missing
    }

# ================= ML LOGIC END =================

app = FastAPI()

def extract_text(file: UploadFile):
    content = file.file.read()
    if file.filename.endswith(".docx"):
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    elif file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        return "".join(page.extract_text() or "" for page in reader.pages)
    else:
        return content.decode("utf-8")

@app.post("/generate-plan")
async def generate_plan(file: UploadFile = File(...)):
    text = extract_text(file)
    return generate_onboarding_plan(text)
