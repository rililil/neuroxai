import base64
import os
import pickle
import json
from io import BytesIO
from functools import wraps
from datetime import datetime
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

app = Flask(__name__)
app.secret_key = "change_this_secret_key_to_something_more_secure"

USERS_DB_PATH = "users_auth.json"
PATIENTS_DB_PATH = "patients.db"

# Initialize SQLite database for patient records
def init_db():
    conn = sqlite3.connect(PATIENTS_DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_id TEXT NOT NULL,
                  patient_name TEXT NOT NULL,
                  patient_age INTEGER NOT NULL,
                  doctor_name TEXT NOT NULL,
                  prediction_label TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  explanation TEXT NOT NULL,
                  features TEXT NOT NULL,
                  doctor_feedback TEXT,
                  corrected_diagnosis TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

def load_users():
    if not os.path.exists(USERS_DB_PATH):
        return {}
    with open(USERS_DB_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def save_users(users):
    with open(USERS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped

ARTIFACTS_PATH = os.path.join("artifacts", "model_artifacts.pkl")

if not os.path.exists(ARTIFACTS_PATH):
    raise FileNotFoundError(f"Model artifacts not found at {ARTIFACTS_PATH}. Please run train_model.py first.")

with open(ARTIFACTS_PATH, "rb") as f:
    artifacts = pickle.load(f)

pipeline = artifacts["pipeline"]
feature_names = artifacts["feature_names"]
feature_stats = artifacts["feature_stats"]
label_map = artifacts["label_map"]

DISPLAY_NAMES = {
    "total_time23": "Total Writing Time (Task 23)",
    "total_time17": "Total Writing Time (Task 17)",
    "total_time15": "Total Writing Time (Task 15)",
    "air_time17": "Air Movement Time (Task 17)",
    "paper_time23": "On-Paper Time (Task 23)",
    "air_time23": "Air Movement Time (Task 23)",
    "air_time15": "Air Movement Time (Task 15)",
    "total_time13": "Total Writing Time (Task 13)",
    "mean_speed_in_air17": "Average Air Speed (Task 17)",
    "paper_time17": "On-Paper Time (Task 17)",
    "total_time9": "Total Writing Time (Task 9)",
    "total_time6": "Total Writing Time (Task 6)",
    "total_time16": "Total Writing Time (Task 16)",
    "mean_acc_in_air17": "Average Air Acceleration (Task 17)",
    "gmrt_in_air17": "Movement Irregularity (GMRT, T17)",
    "total_time8": "Total Writing Time (Task 8)",
    "mean_gmrt17": "Average GMRT (Task 17)",
    "air_time22": "Air Movement Time (Task 22)",
    "mean_jerk_in_air17": "Average Jerk in Air (Task 17)",
    "total_time2": "Total Writing Time (Task 2)"
}

scaler = pipeline.named_steps["scaler"]
model = pipeline.named_steps["model"]

explainer = shap.TreeExplainer(model)
THRESHOLD = 0.6

@app.route("/")
def welcome():
    if "doctor_name" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    doctor = session.get("doctor_name", "Doctor")
    
    # Get statistics
    conn = sqlite3.connect(PATIENTS_DB_PATH)
    c = conn.cursor()
    
    # Total patients
    c.execute("SELECT COUNT(*) FROM patients WHERE doctor_name = ?", (doctor,))
    total_patients = c.fetchone()[0]
    
    # Patients today
    c.execute("SELECT COUNT(*) FROM patients WHERE doctor_name = ? AND DATE(created_at) = DATE('now')", (doctor,))
    patients_today = c.fetchone()[0]
    
    # Healthy vs Patient count
    c.execute("SELECT prediction_label, COUNT(*) FROM patients WHERE doctor_name = ? GROUP BY prediction_label", (doctor,))
    diagnosis_counts = dict(c.fetchall())
    
    # Recent patients
    c.execute("SELECT patient_id, patient_name, patient_age, prediction_label, confidence, created_at FROM patients WHERE doctor_name = ? ORDER BY created_at DESC LIMIT 5", (doctor,))
    recent_patients = c.fetchall()
    
    conn.close()
    
    return render_template("dashboard.html", 
                         doctor=doctor,
                         total_patients=total_patients,
                         patients_today=patients_today,
                         diagnosis_counts=diagnosis_counts,
                         recent_patients=recent_patients)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        username = request.form.get("username", "").strip()
        phone = request.form.get("phone", "").strip()
        password = request.form.get("password", "")

        users = load_users()

        if not email or not username or not password or not phone:
            error = "Email, username and password are required."
        elif username in users:
            error = "This username is already taken."
        else:
            for u_data in users.values():
                if isinstance(u_data, dict):
                    if u_data.get("email", "").lower() == email.lower():
                        error = "This email is already registered."
                        break

        if not error:
            pwd_hash = generate_password_hash(password)
            users[username] = {"email": email,"phone":phone, "password": pwd_hash}
            save_users(users)
            flash("Account created successfully. Please log in.", "success")
            return redirect(url_for("login"))

    return render_template("signup.html", error=error)

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username_or_email = request.form.get("username_or_email", "").strip()
        password = request.form.get("password", "")

        users = load_users()

        username = None
        pwd_hash = None

        if username_or_email in users:
            data = users[username_or_email]
            username = username_or_email
            pwd_hash = data.get("password") if isinstance(data, dict) else data
        else:
            for uname, data in users.items():
                if isinstance(data, dict) :
                    if(
                       data.get("email", "").lower() == username_or_email.lower()
                       or data.get("phone", "") == username_or_email
                       ):
                       username = uname
                    pwd_hash = data.get("password")
                    break

        if not pwd_hash or not check_password_hash(pwd_hash, password):
            error = "Invalid username/email or password."
        else:
            session["logged_in"] = True
            session["doctor_name"] = username
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))

    return render_template("login.html", error=error)

@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route("/diagnosis", methods=["GET"])
@login_required
def index():
    return render_template("index.html", 
                         feature_names=feature_names, 
                         feature_stats=feature_stats, 
                         display_names=DISPLAY_NAMES)

@app.route("/patient-history")
@login_required
def patient_history():
    doctor = session.get("doctor_name", "Doctor")
    
    conn = sqlite3.connect(PATIENTS_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, patient_id, patient_name, patient_age, prediction_label, confidence, created_at, doctor_feedback FROM patients WHERE doctor_name = ? ORDER BY created_at DESC", (doctor,))
    patients = c.fetchall()
    conn.close()
    
    return render_template("patient_history.html", patients=patients)

@app.route("/patient-detail/<int:patient_id>")
@login_required
def patient_detail(patient_id):
    doctor = session.get("doctor_name", "Doctor")
    
    conn = sqlite3.connect(PATIENTS_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE id = ? AND doctor_name = ?", (patient_id, doctor))
    patient = c.fetchone()
    conn.close()
    
    if not patient:
        flash("Patient record not found.", "error")
        return redirect(url_for("patient_history"))
    
    # Parse features JSON
    features = json.loads(patient[8])
    
    return render_template("patient_detail.html", 
                         patient=patient,
                         features=features,
                         display_names=DISPLAY_NAMES)

def make_shap_plot(shap_values_for_sample, feature_names):
    shap_values_for_sample = np.asarray(shap_values_for_sample).reshape(-1)
    shap_series = pd.Series(shap_values_for_sample, index=feature_names)
    shap_abs = shap_series.abs().sort_values(ascending=False)
    top_features = shap_abs.head(10).index
    shap_top = shap_series[top_features]

    plt.figure(figsize=(10, 6))
    colors_list = ['#2ecc71' if x > 0 else '#0066cc' for x in shap_top.sort_values()]
    shap_top.sort_values().plot(kind="barh", color=colors_list)
    plt.xlabel("SHAP Value (Impact on Prediction)", fontsize=12, fontweight='bold')
    plt.title("Top 10 Features Contributing to Prediction", fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def make_comparison_plot(patient_values, feature_names, feature_stats):
    """Create radar chart comparing patient with healthy average"""
    # Select top 8 features for cleaner visualization
    top_features = feature_names[:8]
    
    # Normalize values to 0-1 scale for comparison
    patient_normalized = []
    healthy_normalized = []
    
    for feat in top_features:
        stats = feature_stats[feat]
        patient_val = patient_values[feature_names.index(feat)]
        
        # Normalize
        feat_range = stats['max'] - stats['min']
        if feat_range > 0:
            patient_norm = (patient_val - stats['min']) / feat_range
            healthy_norm = (stats['mean'] - stats['min']) / feat_range
        else:
            patient_norm = 0.5
            healthy_norm = 0.5
            
        patient_normalized.append(patient_norm)
        healthy_normalized.append(healthy_norm)
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(top_features), endpoint=False).tolist()
    patient_normalized += patient_normalized[:1]
    healthy_normalized += healthy_normalized[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, patient_normalized, 'o-', linewidth=2, label='Patient', color='#e74c3c')
    ax.fill(angles, patient_normalized, alpha=0.25, color='#e74c3c')
    ax.plot(angles, healthy_normalized, 'o-', linewidth=2, label='Healthy Average', color='#0066cc')
    ax.fill(angles, healthy_normalized, alpha=0.25, color='#0066cc')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([DISPLAY_NAMES.get(f, f)[:30] for f in top_features], size=9)
    ax.set_ylim(0, 1)
    ax.set_title("Patient vs Healthy Population Comparison", size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    doctor = session.get("doctor_name", "Doctor")
    
    # Patient info
    patient_id = request.form.get("patient_id", "N/A")
    patient_name = request.form.get("patient_name", "Unknown")
    patient_age = request.form.get("patient_age", "N/A")

    # Read feature values
    values = []
    for feat in feature_names:
        val_str = request.form.get(feat)
        stats = feature_stats.get(feat, {"mean": 0})
        if val_str is None or val_str == "":
            val = stats.get("mean", 0)
        else:
            try:
                val = float(val_str)
            except ValueError:
                val = stats.get("mean", 0)
        values.append(val)

    input_df = pd.DataFrame([values], columns=feature_names)

    # Prediction
    try:
        proba_patient = pipeline.predict_proba(input_df)[0, 1]
    except Exception as e:
        print("predict_proba error:", e)
        proba_patient = 0.5

    pred_class = int(proba_patient >= THRESHOLD)
    label_text = label_map.get(pred_class, str(pred_class))
    confidence = proba_patient if pred_class == 1 else (1.0 - proba_patient)

    # Text explanation
    diffs = {}
    for feat, val in zip(feature_names, values):
        stats = feature_stats.get(feat, {"mean": 0})
        mean_val = stats.get("mean", 0)
        diffs[feat] = abs(val - mean_val)

    sorted_feats = sorted(diffs.items(), key=lambda x: x[1], reverse=True)
    top_feats = [f for f, _ in sorted_feats[:3]]

    reasons = []
    for feat in top_feats:
        stats = feature_stats.get(feat, {"mean": 0})
        mean_val = stats.get("mean", 0)
        val = input_df.iloc[0][feat]
        level = "higher than average" if val >= mean_val else "lower than average"
        nice_name = DISPLAY_NAMES.get(feat, feat)
        reasons.append(f"{nice_name} was {level}")

    explanation_text = (
        f"Result: {label_text} ({confidence * 100:.1f}% confidence). "
        f"Key factors: {', '.join(reasons)}."
    )

    # SHAP plot
    shap_plot_base64 = ""
    try:
        input_scaled = scaler.transform(input_df)
        shap_values = explainer.shap_values(input_scaled)

        sv = shap_values
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv)

        if sv.ndim == 3:
            idx_class = -1 if sv.shape[2] > 1 else 0
            shap_for_sample = sv[0, :, idx_class]
        elif sv.ndim == 2:
            shap_for_sample = sv[0, :]
        else:
            shap_for_sample = sv.reshape(-1)

        shap_plot_base64 = make_shap_plot(shap_for_sample, feature_names)
    except Exception as e:
        print("SHAP error:", e)
        shap_plot_base64 = ""
    
    # Comparison plot
    comparison_plot_base64 = ""
    try:
        comparison_plot_base64 = make_comparison_plot(values, feature_names, feature_stats)
    except Exception as e:
        print("Comparison plot error:", e)

    # Save to database
    conn = sqlite3.connect(PATIENTS_DB_PATH)
    c = conn.cursor()
    features_json = json.dumps(dict(zip(feature_names, values)))
    c.execute("""INSERT INTO patients 
                 (patient_id, patient_name, patient_age, doctor_name, prediction_label, confidence, explanation, features)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
              (patient_id, patient_name, patient_age, doctor, label_text, confidence * 100.0, explanation_text, features_json))
    db_patient_id = c.lastrowid
    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        patient_id=patient_id,
        db_patient_id=db_patient_id,
        prediction_label=label_text,
        probability=confidence * 100.0,
        explanation_text=explanation_text,
        shap_plot=shap_plot_base64,
        comparison_plot=comparison_plot_base64,
        patient_name=patient_name,
        patient_age=patient_age,
    )

@app.route("/submit-feedback/<int:patient_id>", methods=["POST"])
@login_required
def submit_feedback(patient_id):
    doctor = session.get("doctor_name", "Doctor")
    feedback = request.form.get("feedback", "").strip()
    corrected_diagnosis = request.form.get("corrected_diagnosis", "").strip()
    
    conn = sqlite3.connect(PATIENTS_DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE patients SET doctor_feedback = ?, corrected_diagnosis = ? WHERE id = ? AND doctor_name = ?",
              (feedback, corrected_diagnosis, patient_id, doctor))
    conn.commit()
    conn.close()
    
    flash("Feedback submitted successfully!", "success")
    return redirect(url_for("patient_detail", patient_id=patient_id))

@app.route("/export-pdf/<int:patient_id>")
@login_required
def export_pdf(patient_id):
    doctor = session.get("doctor_name", "Doctor")
    
    conn = sqlite3.connect(PATIENTS_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE id = ? AND doctor_name = ?", (patient_id, doctor))
    patient = c.fetchone()
    conn.close()
    
    if not patient:
        flash("Patient record not found.", "error")
        return redirect(url_for("patient_history"))
    
    # Create PDF
    pdf_filename = f"patient_report_{patient_id}.pdf"
    pdf_path = os.path.join("/tmp", pdf_filename)
    
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("NeuroXAI Care - Patient Diagnostic Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Patient Information
    story.append(Paragraph("Patient Information", heading_style))
    patient_data = [
        ['Patient ID:', patient[1]],
        ['Patient Name:', patient[2]],
        ['Patient Age:', str(patient[3])],
        ['Doctor:', patient[4]],
        ['Date:', patient[11]]
    ]
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Diagnosis
    story.append(Paragraph("Diagnosis Results", heading_style))
    diagnosis_color = colors.HexColor('#e74c3c') if patient[5] == 'Patient' else colors.HexColor('#2ecc71')
    diagnosis_data = [
        ['Diagnosis:', patient[5]],
        ['Confidence:', f"{patient[6]:.1f}%"],
    ]
    diagnosis_table = Table(diagnosis_data, colWidths=[2*inch, 4*inch])
    diagnosis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('BACKGROUND', (1, 0), (1, 0), diagnosis_color),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (1, 0), (1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
    ]))
    story.append(diagnosis_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Explanation
    story.append(Paragraph("Clinical Explanation", heading_style))
    story.append(Paragraph(patient[7], styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Doctor's Feedback
    if patient[9]:
        story.append(Paragraph("Doctor's Feedback", heading_style))
        story.append(Paragraph(patient[9], styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    if patient[10]:
        story.append(Paragraph("Corrected Diagnosis", heading_style))
        story.append(Paragraph(patient[10], styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
    
    # Footer
    story.append(Spacer(1, 0.5*inch))
    footer_text = "This report is generated by NeuroXAI Care Clinical Decision Support System. For medical use only."
    story.append(Paragraph(footer_text, ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)))
    
    doc.build(story)
    
    return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)

@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/patient/login", methods=["GET", "POST"])
def patient_login():
    if request.method == "POST":
        patient_id = request.form["patient_id"]
        patient_name = request.form["patient_name"]

        conn = sqlite3.connect(PATIENTS_DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM patients
            WHERE patient_id=? AND patient_name=?
            ORDER BY created_at DESC
            LIMIT 1
        """, (patient_id, patient_name))

        record = cursor.fetchone()
        conn.close()

        if record:
            session["patient_logged_in"] = True
            session["patient_id"] = patient_id
            return redirect("/patient/dashboard")
        else:
            return render_template("patient_login.html", error="No record found.")

    return render_template("patient_login.html")
@app.route("/patient/dashboard")
def patient_dashboard():
    if not session.get("patient_logged_in"):
        return redirect("/patient/login")

    patient_id = session.get("patient_id")

    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM patients
        WHERE patient_id=?
        ORDER BY created_at DESC
        LIMIT 1
    """, (patient_id,))

    record = cursor.fetchone()
    conn.close()

    if not record:
        return redirect("/patient/login")

    shap_plot_base64 = ""
    comparison_plot_base64 = ""

    try:
        # نفس طريقة الدكتور بالضبط
        features_dict = json.loads(record[8])  # عمود features
        values = [features_dict[f] for f in feature_names]

        input_df = pd.DataFrame([values], columns=feature_names)

        input_scaled = scaler.transform(input_df)
        shap_values = explainer.shap_values(input_scaled)

        sv = shap_values
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv)

        if sv.ndim == 3:
            idx_class = -1 if sv.shape[2] > 1 else 0
            shap_for_sample = sv[0, :, idx_class]
        elif sv.ndim == 2:
            shap_for_sample = sv[0, :]
        else:
            shap_for_sample = sv.reshape(-1)

        shap_plot_base64 = make_shap_plot(shap_for_sample, feature_names)
        comparison_plot_base64 = make_comparison_plot(values, feature_names, feature_stats)

    except Exception as e:
        print("Patient chart error:", e)

    return render_template(
        "patient_dashboard.html",
        record=record,
        shap_plot=shap_plot_base64,
        comparison_plot=comparison_plot_base64
    )
if __name__ == "__main__":
    import webbrowser
    from threading import Timer
    
    def open_browser():
        webbrowser.open('http://127.0.0.1:5000/login')
    
    Timer(1.5, open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=False)

