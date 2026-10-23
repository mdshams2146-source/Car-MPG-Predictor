# Car MPG Predictor 🚗

A clean, modern, and professional Streamlit interface for predicting vehicle fuel efficiency (Miles Per Gallon) using a pre-trained Machine Learning model (`mpg.h5`) with integrated SQLite prediction history.

## 🌟 Features

- **ML Prediction Model**: Loads pre-trained model `mpg.h5` using `joblib.load("mpg.h5")`.
- **7-Feature Input Vector**: Expects `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, `model_year`, `origin` in exact order.
- **User-Friendly Interface**: Responsive 2-column input design with clear selections (e.g. Origin mapped to numeric values).
- **Automotive Aesthetic**: Modern glassmorphism dashboard styling.
- **SQLite History Storage**: Saves prediction history (timestamp, input parameters, predicted MPG) locally in `history.db`.
- **Input Validation**: Ensures valid, non-zero positive numbers.

## 📁 Project Structure

```
MPG-Predictor/
│
├── mpg.h5              # Pre-trained ML Linear Regression Model
├── app.py              # Streamlit Application Code
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

## 📊 Model Features Specification

| Feature | Description | Example / Range |
| :--- | :--- | :--- |
| **Cylinders** | Number of engine cylinders | 3, 4, 5, 6, 8 |
| **Displacement** | Engine displacement (cu. in.) | 50.0 - 500.0 |
| **Horsepower** | Power output (hp) | 40.0 - 350.0 |
| **Weight** | Vehicle curb weight (lbs) | 1500 - 6000 |
| **Acceleration** | 0-60 mph time (seconds) | 5.0 - 30.0 |
| **Model Year** | Year of manufacture | 70 - 82 (e.g. 1978 = 78) |
| **Origin** | Car origin | USA (1), Europe (2), Japan (3) |

## 🗄️ SQLite Database History
Predictions are automatically saved to `history.db` under the `prediction_history` table. You can inspect or clear past predictions directly from the collapsible **"View Prediction History"** drawer in the UI.
