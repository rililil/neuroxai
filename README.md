# NeuroXAI Care - Clinical Decision Support System

## 🧠 Overview
NeuroXAI Care is an advanced AI-powered clinical decision support system for early detection of neurodegenerative diseases using handwriting analysis. The system leverages Machine Learning and Explainable AI (XAI) to provide doctors with transparent, interpretable diagnostic insights.

## ✨ Features

### 1. **Enhanced Dashboard**
- Real-time statistics display
- Patient assessment tracking
- Visual analytics for diagnosis trends
- Quick access to recent assessments

### 2. **Patient History Management**
- Complete patient records database
- Search and filter functionality
- Track assessment history
- Monitor patient progression over time

### 3. **Advanced Diagnostic Analysis**
- SHAP (SHapley Additive exPlanations) visualization
- Radar chart comparison with healthy population
- Confidence scoring
- Detailed feature analysis

### 4. **Doctor Feedback System**
- Human-in-the-loop capability
- Corrected diagnosis tracking
- Clinical notes integration
- Continuous model improvement

### 5. **PDF Report Generation**
- Professional clinical reports
- Export patient data
- Include all diagnostic visualizations
- Print-ready format

### 6. **Modern UI/UX**
- Responsive design
- Dark/Light mode toggle
- Intuitive navigation
- Professional medical interface

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Prepare the model:**
```bash
python train_model.py
```
This will create the model artifacts in the `artifacts/` folder.

3. **Run the application:**
```bash
python app.py
```

4. **Access the system:**
Open your browser and navigate to: `http://localhost:5000`

## 📁 Project Structure

```
NeuroXAI-Care/
│
├── app.py                      # Main Flask application
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── DARWIN.csv                  # Training dataset
│
├── templates/                  # HTML templates
│   ├── dashboard.html          # Main dashboard
│   ├── login.html              # Login page
│   ├── signup.html             # Registration page
│   ├── index.html              # Diagnosis input form
│   ├── result.html             # Results display
│   ├── patient_history.html    # Patient records
│   ├── patient_detail.html     # Detailed patient view
│   └── about.html              # About page
│
├── static/                     # Static files
│   ├── style.css               # Main stylesheet
│   └── L.PNG                   # Logo
│
├── artifacts/                  # Model artifacts
│   └── model_artifacts.pkl     # Trained model
│
└── patients.db                 # SQLite database
```

## 🔐 Authentication System

The system includes a secure authentication mechanism:
- User registration with email validation
- Password hashing (werkzeug security)
- Session management
- Protected routes

### Default Setup
1. Register a new account at `/signup`
2. Login with your credentials at `/login`
3. Access the dashboard and features

## 📊 Database Schema

**Patients Table:**
- `id`: Primary key
- `patient_name`: Patient's name
- `patient_age`: Patient's age
- `doctor_name`: Assigned doctor
- `prediction_label`: AI diagnosis (Healthy/Patient)
- `confidence`: Confidence percentage
- `explanation`: Clinical explanation
- `features`: JSON of handwriting features
- `doctor_feedback`: Doctor's clinical notes
- `corrected_diagnosis`: Corrected diagnosis if AI was wrong
- `created_at`: Timestamp

## 🎯 Usage Guide

### For Doctors:

1. **Login**: Access your account
2. **Dashboard**: View statistics and recent assessments
3. **New Diagnosis**: 
   - Enter patient information
   - Adjust handwriting biomarker sliders
   - Click "Predict Diagnosis"
4. **Review Results**:
   - View AI prediction with confidence score
   - Analyze SHAP explanations
   - Compare with healthy population
5. **Provide Feedback**:
   - Add clinical notes
   - Correct diagnosis if needed
   - Export PDF report
6. **Track History**: Monitor all patient assessments

### For Researchers:

- Review the DARWIN dataset features
- Analyze feature importance
- Study SHAP values for model interpretability
- Export data for further analysis

## 🧪 Model Details

**Algorithm**: Random Forest Classifier
- **Features**: 20 handwriting biomarkers
- **Training**: DARWIN dataset
- **Explainability**: SHAP TreeExplainer
- **Threshold**: 0.6 (adjustable)

### Key Features:
- Total writing time (various tasks)
- Air movement time
- On-paper time
- Speed and acceleration metrics
- Movement irregularity (GMRT)
- Jerk measurements

## 📈 Performance Metrics

The model provides:
- Binary classification (Healthy vs Patient)
- Confidence scores
- Feature importance rankings
- SHAP value explanations

## 🔧 Customization

### Adjust Prediction Threshold:
In `app.py`, modify:
```python
THRESHOLD = 0.6  # Change to desired value (0.0 - 1.0)
```

### Modify Features:
In `train_model.py`, edit the `FEATURE_NAMES` list

### Styling:
Customize colors and design in `static/style.css`

## 🛡️ Security Considerations

- Change the secret key in `app.py`:
```python
app.secret_key = "your_secure_secret_key_here"
```
- Use environment variables for production
- Implement HTTPS in production
- Regular database backups

## 📞 Support & Contact

For questions, issues, or feature requests:
- Email: contact.us@neuroxai.com
- Project Repository: [GitHub Link]

## 📝 License

This project is developed for research and educational purposes.

## 👥 Team

Developed by students from the **College of Computer Science and Engineering**

## 🙏 Acknowledgments

- DARWIN dataset contributors
- SHAP library developers
- Flask framework team
- Medical advisors and consultants

---

**Note**: This system is a clinical decision support tool and should not replace professional medical judgment. Always combine AI insights with clinical expertise and additional diagnostic methods.
