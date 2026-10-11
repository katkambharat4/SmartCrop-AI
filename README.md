# SmartCrop AI – AI-Based Smart Crop Recommendation System

> **Academic Description:**  
> "SmartCrop AI is a supervised machine learning-based crop recommendation system that recommends a suitable crop based on agricultural and environmental conditions."

---

## 1. Project Title
**SmartCrop AI – AI-Based Smart Crop Recommendation System**  
A simple, robust decision-support web application developed as a B.Tech Computer Science project.

---

## 2. Project Objective
The purpose of SmartCrop AI is to help agricultural practitioners and students select a suitable crop before cultivation based on specific agricultural (soil nutrients) and environmental conditions.

The application accepts current/user-provided values and uses a trained **Random Forest Machine Learning classification model** to predict and recommend the most suitable crop.

---

## 3. Problem Statement
Choosing an unsuitable crop for a given soil and climatic profile leads to sub-optimal yields, soil degradation, and economic losses. Traditional methods often rely on guesswork or generalized crop calendars that do not account for variations in macronutrients (Nitrogen, Phosphorus, Potassium) and climate metrics (temperature, relative humidity, rainfall). A data-driven decision-support tool can guide farmers and agronomists toward informed decisions before planting.

---

## 4. Proposed Solution
SmartCrop AI addresses this challenge through a clean, transparent web interface and machine learning classification pipeline:
1. The user enters measured soil and climate values through a web form.
2. The system executes client-side and server-side validation to ensure all inputs are numerically sound.
3. A pre-trained Random Forest Classifier evaluates the parameters against agricultural benchmarks.
4. The system presents the recommended crop with an input summary and decision-support guidance.

---

## 5. System Scope & Disclaimer
- **Classification Problem:** Multi-class supervised classification.
- **Decision-Support System:** Provides empirical recommendations based on data; it is **NOT** a guaranteed farming outcome system.
- **Current/User-Provided Inputs:** Uses user-provided agricultural values. It does **NOT** use real-time sensor data or IoT telemetry in this version.
- **Not Included:** Does not diagnose crop diseases, provide automated irrigation controls, forecast commodity prices, or include chatbots/databases.

---

## 6. Features
- **Clean Agriculture-Themed Web UI:** Built with lightweight, responsive HTML, CSS, and JavaScript.
- **Robust Two-Tier Input Validation:** Client-side JavaScript validation plus strict server-side validation in Python/Flask prevents invalid numbers, out-of-range humidity (>100%), negative rainfall, or missing parameters from crashing the application.
- **Dedicated Preprocessing Pipeline:** Validates and formats user inputs in `utils/preprocessing.py` into the exact feature order expected by the ML model.
- **Reproducible ML Pipeline:** Scripted training pipeline (`train_model.py`) that checks missing data, stratifies train-test splits, reports exact accuracy metrics, and serializes the model using Joblib.
- **No Retraining Overhead:** Inference uses the serialized model (`model/crop_model.pkl`), ensuring sub-second response times.
- **Comprehensive Input Summary:** Result view clearly presents all inputs submitted by the user alongside the recommended crop.

---

## 7. Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | HTML5, CSS3, JavaScript (ES6) | Responsive user interface, form validation, and results presentation |
| **Backend** | Python 3, Flask | Routing, server-side validation, model inference |
| **Machine Learning** | Scikit-learn (Random Forest Classifier) | Supervised multi-class classification model |
| **Data Processing** | Pandas, NumPy | Dataset loading, array reshaping, feature ordering |
| **Model Persistence**| Joblib | Saving and loading trained model binaries |
| **Dataset** | Crop Recommendation Dataset | Benchmark agricultural dataset (2,200 instances) |
| **Database** | *None required* | Stateless web application architecture |

---

## 8. Machine Learning Algorithm
- **Algorithm:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
- **Type:** Supervised Learning (Ensemble of Decision Trees via Bagging)
- **Hyperparameters:** `n_estimators=100`, `random_state=42`
- **Why Random Forest?**
  1. High accuracy on tabular agricultural data with non-linear feature interactions.
  2. Resilient to outliers and overfitting due to bootstrap aggregating and feature subsampling.
  3. Fast inference time suitable for real-time web deployment without requiring GPU hardware.

---

## 9. Dataset Details
- **File Location:** `data/crop_recommendation.csv`
- **Total Records:** 2,200 rows
- **Missing Values:** 0 null values
- **Target Variable (`label`):** 22 unique crop classes:
  - *apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon*
- **Trained Features (6 environmental & soil features):**
  1. `N`: Nitrogen ratio in soil (mg/kg)
  2. `P`: Phosphorus ratio in soil (mg/kg)
  3. `K`: Potassium ratio in soil (mg/kg)
  4. `temperature`: Atmospheric temperature (°C)
  5. `humidity`: Relative humidity (%)
  6. `rainfall`: Rainfall depth (mm)
- **Season Feature Handling:** The user interface collects `Season` (e.g., Kharif, Rabi, Zaid, Summer, Winter, Monsoon) as user-provided context. Because `Season` is not a column in the underlying benchmark dataset, it is not artificially injected into model training. The model is trained strictly on the 6 verified features, preserving scientific validity.

---

## 10. Actual Model Evaluation Metrics
*Trained on 80% train split (1,760 samples) and evaluated on 20% test split (440 samples) with fixed `random_state=42`:*

- **Dataset Size:** 2,200 samples
- **Number of Features:** 6
- **Number of Classes:** 22
- **Training Accuracy:** **100.00%**
- **Testing Accuracy:** **99.09%** (436 out of 440 test samples predicted correctly)

### Test Set Classification Report Summary
- **Macro Average Precision:** 0.99
- **Macro Average Recall:** 0.99
- **Macro Average F1-Score:** 0.99

---

## 11. System Architecture

```
User
  │
  ▼
Frontend (HTML / CSS / JavaScript)
  │
  ▼
Input Validation (Client-Side & Server-Side Bounds Checks)
  │
  ▼
Flask Backend (app.py)
  │
  ▼
Preprocessing & Feature Ordering (utils/preprocessing.py)
  │
  ▼
Random Forest Model (model/crop_model.pkl)
  │
  ▼
Crop Prediction
  │
  ▼
Result Page (templates/result.html)
```

---

## 12. Machine Learning Workflow

```
Dataset (data/crop_recommendation.csv)
  │
  ▼
Data Inspection & Preprocessing (Missing value checks, column alignment)
  │
  ▼
Feature Selection ([N, P, K, temperature, humidity, rainfall])
  │
  ▼
Stratified Train/Test Split (80% Train, 20% Test)
  │
  ▼
Random Forest Training (100 Decision Trees, random_state=42)
  │
  ▼
Model Evaluation (Accuracy: 99.09%, Classification Report)
  │
  ▼
Save Serialized Model (Joblib -> model/crop_model.pkl)
  │
  ▼
User Input via Flask Web Form
  │
  ▼
Prediction Inference (model.predict)
  │
  ▼
Recommended Crop & Input Summary Displayed
```

---

## 13. Project Structure

```
SmartCrop-AI/
├── app.py                      # Flask web server and routing
├── train_model.py              # ML training and evaluation script
├── test_app.py                 # Automated unit and integration test suite
├── requirements.txt            # Python dependencies
├── README.md                   # Complete project documentation
│
├── data/
│   └── crop_recommendation.csv # Real agricultural benchmark dataset (2200 rows)
│
├── model/
│   ├── crop_model.pkl          # Serialized Random Forest Classifier
│   └── model_metadata.pkl      # Saved classes, feature list, and accuracy records
│
├── templates/
│   ├── base.html               # Shared layout template with navbar and footer
│   ├── index.html              # Home page with project overview
│   ├── predict.html            # Prediction input form with validation hints
│   └── result.html             # Recommendation result & input summary view
│
├── static/
│   ├── css/
│   │   └── style.css           # Responsive green & white agricultural stylesheet
│   └── js/
│       └── script.js           # Client-side input validation and UX
│
└── utils/
    └── preprocessing.py        # Input data validation and feature ordering logic
```

---

## 14. Installation & Windows Setup

### Step 1: Open PowerShell
Open Windows PowerShell and navigate to the project directory:
```powershell
cd c:\Users\Lenovo\OneDrive\Desktop\SmartCrop-AI
```

### Step 2: Create Virtual Environment
```powershell
python -m venv venv
```

### Step 3: Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
*Note for Windows:* If PowerShell displays an `Execution_Policies` restriction error, you can bypass the policy for the current process:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
```
Or run python directly via the virtual environment path:
```powershell
.\venv\Scripts\python.exe <command>
```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 15. Training the Model
To inspect the dataset, calculate metrics, and save the model:
```powershell
python train_model.py
```
Expected output:
- Dataset shape: `2200 rows, 8 columns`
- Missing values: `0`
- Class count: `22 crops`
- Training accuracy: `100.00%`
- Testing accuracy: `99.09%`
- Saved model: `model/crop_model.pkl`

---

## 16. Running the Web Application
Start the Flask development server:
```powershell
python app.py
```
Access the application in your browser at:
```
http://127.0.0.1:5000/
```

To run the automated test suite:
```powershell
python test_app.py
```

---

## 17. Example Usage

### Example 1: Rice Conditions
- **Nitrogen (N):** `90`
- **Phosphorus (P):** `42`
- **Potassium (K):** `43`
- **Temperature:** `20.87` °C
- **Relative Humidity:** `82.00` %
- **Rainfall:** `202.93` mm
- **Season:** `Kharif`
- **Predicted Crop:** **Rice**

### Example 2: Coffee Conditions
- **Nitrogen (N):** `104`
- **Phosphorus (P):** `18`
- **Potassium (K):** `30`
- **Temperature:** `26.50` °C
- **Relative Humidity:** `58.00` %
- **Rainfall:** `160.00` mm
- **Season:** `Winter`
- **Predicted Crop:** **Coffee**

---

## 18. Limitations
1. **Empirical Static Dataset:** Recommendations are based on patterns in the 2,200 benchmark instances; microclimatic anomalies may not be captured.
2. **User-Provided Inputs:** Requires manual data entry by the user rather than automated live sensors.
3. **No Economic Factors:** Does not account for dynamic market prices, fertilizer costs, or transport logistics.
4. **Advisory Nature:** Serves as a decision-support guide and should be corroborated with local agricultural extension expertise.

---

## 19. Future Enhancements
The following features are identified as candidate future extensions:
- Integration with live weather APIs (e.g., OpenWeatherMap) for automatic climatic parameters.
- IoT soil moisture, pH, and NPK sensor hardware integration.
- Dedicated mobile application (Android / iOS).
- Crop yield estimation based on historical acreage and fertilizer dosage.
- Soil moisture monitoring and irrigation planning.
- Regional and state-level crop zoning guidelines.

---

## 20. Viva / Interview Questions & Answers

**Q1: What is the core objective of SmartCrop AI?**  
*Answer:* It is a decision-support tool that uses supervised machine learning (Random Forest Classifier) to recommend the most suitable crop to cultivate based on soil nutrients (N, P, K) and climatic parameters (temperature, humidity, rainfall).

**Q2: Why did you choose Random Forest over a single Decision Tree?**  
*Answer:* A single Decision Tree is prone to high variance and overfitting. Random Forest builds an ensemble of 100 decision trees trained on bootstrap samples and aggregates their predictions, resulting in lower variance, higher test accuracy (99.09%), and resilience to noisy data.

**Q3: How are inputs validated before prediction?**  
*Answer:* We implemented dual-tier validation. The client-side JavaScript verifies numeric types and immediate bounds. The Flask backend (`utils/preprocessing.py`) rigorously validates that N, P, K, and rainfall are non-negative, humidity is bounded strictly between 0 and 100%, and numbers are parseable floats before feeding them to the model.

**Q4: Why isn't 'Season' used as an ML training feature?**  
*Answer:* In accordance with scientific integrity and project guidelines, we inspected the authentic dataset (`data/crop_recommendation.csv`) and observed that 'Season' was not present. Fabricating a season column would invalidate the benchmark dataset. Hence, 'Season' is captured for user context and displayed in the result summary, while the ML model is trained strictly on the 6 verified features.

**Q5: What happens if the trained model file is missing when the user clicks predict?**  
*Answer:* The Flask backend catches missing model dependencies gracefully and returns an HTTP 503 response with a clear user-facing error message prompting the administrator to run `train_model.py`, preventing application crashes or raw stack traces.
