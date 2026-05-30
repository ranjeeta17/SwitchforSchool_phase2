# AI-Powered Emotional Intelligence Support — Switch4Schools
**IFN735 Industry Project | Team 29 | Phase 1 & Phase 2**

> **Author of project structure & src modules:** Quantum Hustle Team
> Supervisors: Venkat Venkatachalam

# How to run the project
pip install -r requirements.txt
python run_pipeline.py         # Phase 1 + Phase 2 analysis
streamlit run app/app.py       # Teacher dashboard
---

## 📁 Project Structure

```
emotional_intelligence_project/
│
├── 📂 datasets/                          # Raw & cleaned CSV data from Switch4Schools
│   ├── studentCheckIn_cleaned.csv        # 757,112 check-in records (primary dataset)
│   ├── checkInSession.csv                # 31,149 sessions
│   ├── class.csv                         # 583 class definitions
│   ├── students.csv                      # 12,467 students
│   ├── teacherClass.csv                  # 3,754 teacher-class mappings
│   ├── teacherCheckIn.csv                # Staff check-in data
│   ├── wellbeingResponse.csv             # Survey responses (~72 MB)
│   ├── wellbeingQuestion.csv             # Survey schema
│   └── switches-summary.csv             # Activity/intervention catalogue
│
├── 📂 outputs/                           # Generated results (auto-created by pipeline)
│   ├── class_emotional_profiles.json     # 97 class profiles (Phase 1)
│   ├── class_recommendations.json        # Activity recommendations per class (Phase 1)
│   ├── class_analysis_summary.csv        # Flat CSV summary (Phase 1)
│   ├── risk_predictions.json             # Trajectory predictions (Phase 2)
│   └── intervention_log.json             # Before/after outcome log (Phase 2)
│
├── 📂 src/                               # Modular Python source code [@Ranjeeta]
│   ├── __init__.py
│   │
│   ├── 📂 data/                          # Data ingestion & preparation
│   │   ├── __init__.py
│   │   ├── loader.py                     # Centralised CSV loaders for all datasets
│   │   └── preprocessor.py              # Cleaning, merging, normalisation
│   │
│   ├── 📂 profiling/                     # Emotional profile builders
│   │   ├── __init__.py
│   │   ├── class_profiler.py            # Per-class emotional profiles (from main_updated.ipynb)
│   │   └── student_profiler.py          # Per-student emotional profiles (from recomandation.ipynb)
│   │
│   ├── 📂 risk/                          # Risk classification & prediction
│   │   ├── __init__.py
│   │   ├── risk_classifier.py           # Phase 1 — rule-based Low/Medium/High classification
│   │   └── risk_predictor.py            # Phase 2 — Risk Velocity & Escalation Probability Score
│   │
│   ├── 📂 recommendations/              # Recommendation & prioritisation engine
│   │   ├── __init__.py
│   │   ├── activity_recommender.py      # Emotion-to-activity rule engine (from main_updated.ipynb)
│   │   ├── intervention_tracker.py      # Phase 2 — before/after effectiveness tracking
│   │   └── teacher_prioritiser.py       # Phase 2 — composite Priority Score for teacher triage
│   │
│   └── 📂 dashboard/                    # Visualisation & export
│       ├── __init__.py
│       ├── teacher_dashboard.py         # 6-panel matplotlib teacher dashboard
│       └── report_exporter.py           # JSON, CSV export logic
│
├── 📂 notebooks/                         # Jupyter notebooks (exploration & demos)
│   ├── 01_data_exploration.ipynb         # (rename from main.ipynb)
│   ├── 02_phase1_class_profiles.ipynb    # (rename from main_updated.ipynb)
│   └── 03_phase1_student_risk.ipynb      # (rename from recomandation.ipynb)
│
├── 📂 app/                               # Streamlit web dashboard [@Quantum Hustle]
│   └── app.py                            # Interactive teacher-facing UI
│
├── 📂 tests/                             # Unit tests
│   ├── __init__.py
│   └── test_risk_classifier.py          # Tests for rule-based risk logic
│
├── run_pipeline.py                       # 🚀 End-to-end Phase 1 + Phase 2 runner [@Ranjeeta]
├── requirements.txt                      # All Python dependencies
└── README.md                             # This file
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (Phase 1 + Phase 2)
```bash
python run_pipeline.py
```

### 3. Launch the teacher dashboard
```bash
streamlit run app/app.py
```

### 4. Run unit tests
```bash
pytest tests/ -v
```

---

## 📊 What the Pipeline Does

### Phase 1 — Wellbeing Monitoring
| Step | Module | Output |
|------|--------|--------|
| Load all datasets | `src/data/loader.py` | 8 dataframes |
| Clean check-ins | `src/data/preprocessor.py` | 757,112 clean rows |
| Merge relationships | `src/data/preprocessor.py` | 757,112 × 21 table |
| Build class profiles | `src/profiling/class_profiler.py` | 97 class profiles |
| Build student profiles | `src/profiling/student_profiler.py` | 12,467 student profiles |
| Classify risk | `src/risk/risk_classifier.py` | Low / Medium / High per student |
| Generate recommendations | `src/recommendations/activity_recommender.py` | Activity plans per class |
| Export | `src/dashboard/report_exporter.py` | JSON + CSV outputs |

### Phase 2 — Predictive Intelligence
| Step | Module | Output |
|------|--------|--------|
| Risk trajectory prediction | `src/risk/risk_predictor.py` | Escalation Probability Score (0–100) |
| Intervention effectiveness | `src/recommendations/intervention_tracker.py` | Before/after outcome log |
| Teacher priority ranking | `src/recommendations/teacher_prioritiser.py` | Ranked "who to see today" list |

---

## 🧩 Data Schema Reference

### Class Profile (`outputs/class_emotional_profiles.json`)
```json
{
  "class_id": "476f8ef0-...",
  "class_name": "Class 2Q",
  "total_sessions": 159,
  "total_students": 36,
  "emotions": {
    "dominant_emotion": "HAPPY",
    "avg_intensity": 0.75,
    "emotion_distribution": { "HAPPY": 2332, "EXCITED": 1107, ... },
    "recent_emotions": ["HAPPY", "HAPPY", "EXCITED", ...]
  },
  "wellness_indicators": {
    "avg_tiredness": 4.35,
    "chat_requests": 0,
    "absence_rate": 0.016
  },
  "trend_analysis": {
    "trend": "improving",
    "recent_avg_intensity": 0.794,
    "earlier_avg_intensity": 0.706,
    "volatility": 0.256
  },
  "last_session": "2025-03-07T04:15:51Z"
}
```

### Risk Prediction (`outputs/risk_predictions.json`) — Phase 2
```json
{
  "student_id": "...",
  "risk_level": "HIGH",
  "risk_score": 75,
  "risk_velocity": 1.23,
  "escalation_probability": 88.5,
  "trajectory": "escalating",
  "priority_score": 82.1,
  "suggested_action": "Immediate 1-on-1 wellbeing check-in today"
}
```

---

## 👥 Team

| Name | Student No. |
|------|-------------|
| Martin Fahy | N12233510 |
| Makizharasu Mohankumar | N11861088 |
| Jaeyi Lee | N12152773 |
| **Ranjeeta** | **N12166634** |
| Wen-Chi Tzung | N11865181 |

---

## 📚 References
- Beck, K. (2001). *Manifesto for Agile Software Development*
- Dineen et al. (2022). *SEB data use in school districts*
- Dumas et al. (2018). *Fundamentals of Business Process Management* (BPMN 2.0)
- AI-Driven Emotion Analysis Model for Early Detection of Risky Behaviours (2021)
