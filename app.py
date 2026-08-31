import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sqlite3
import os
import base64
import time
import warnings
from datetime import datetime

# Suppress version/feature warnings cleanly
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING SETUP
# ==============================================================================
st.set_page_config(
    page_title="CAR MPG PREDICTOR | Automotive Cockpit ML",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_css(css_file="style.css"):
    """Inject custom CSS stylesheet."""
    if os.path.exists(css_file):
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

def get_base64_image(image_path):
    """Encode local image file to base64 string for background embedding."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

# ==============================================================================
# 2. DATABASE INITIALIZATION (SQLite)
# ==============================================================================
DB_FILE = "history.db"

def init_db():
    """Initialize SQLite database for prediction logging."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cylinders INTEGER NOT NULL,
                displacement REAL NOT NULL,
                horsepower REAL NOT NULL,
                weight REAL NOT NULL,
                acceleration REAL NOT NULL,
                model_year INTEGER NOT NULL,
                origin TEXT NOT NULL,
                predicted_mpg REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    except Exception:
        pass

def save_prediction(cylinders, displacement, horsepower, weight, acceleration, model_year, origin_str, predicted_mpg):
    """Save prediction entry to SQLite database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO prediction_history (timestamp, cylinders, displacement, horsepower, weight, acceleration, model_year, origin, predicted_mpg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now, int(cylinders), float(displacement), float(horsepower), float(weight), float(acceleration), int(model_year), origin_str, round(float(predicted_mpg), 2)))
        conn.commit()
        conn.close()
    except Exception:
        pass

def get_prediction_history():
    """Fetch recent prediction history."""
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("""
            SELECT 
                id as ID, 
                timestamp as Timestamp, 
                cylinders as Cylinders, 
                displacement as 'Displacement', 
                horsepower as 'Horsepower', 
                weight as 'Weight', 
                acceleration as 'Acceleration', 
                model_year as 'Model Year', 
                origin as Origin, 
                predicted_mpg as 'Predicted MPG' 
            FROM prediction_history 
            ORDER BY id DESC 
            LIMIT 50
        """, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def clear_prediction_history():
    """Clear history table."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prediction_history")
        conn.commit()
        conn.close()
    except Exception:
        pass

init_db()

# ==============================================================================
# 3. MODEL LOADING (JOB LIB)
# ==============================================================================
MODEL_PATH = "mpg.h5"

@st.cache_resource
def load_mpg_model():
    """Load existing Linear Regression model."""
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file '{MODEL_PATH}' missing."
    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)

model, model_err = load_mpg_model()

# ==============================================================================
# 4. CINEMATIC HERO SECTION
# ==============================================================================
hero_b64 = get_base64_image(os.path.join("assets", "hero_car.jpg"))
hero_bg_css = f"background-image: url('data:image/jpeg;base64,{hero_b64}');" if hero_b64 else ""

hero_html = f'<div class="hero-card"><div class="hero-overlay"><div class="hero-content"><div class="hero-tag">⚡ AUTOMOTIVE COCKPIT ML</div><h1 class="hero-title-main">CAR MPG</h1><h1 class="hero-title-accent">PREDICTOR</h1><div class="hero-subtitle">Machine Learning <span class="hero-subtitle-dot">•</span> Fuel Efficiency</div></div><div class="hero-image-container" style="{hero_bg_css}"></div></div></div>'
st.markdown(hero_html, unsafe_allow_html=True)

if model_err:
    st.error(f"🚨 Model Error: {model_err}")
    st.stop()

# ==============================================================================
# 5. INPUT CONTROLS & SESSION STATE
# ==============================================================================
DEFAULT_CYLINDERS = 4
DEFAULT_DISPLACEMENT = 150.0
DEFAULT_HORSEPOWER = 95.0
DEFAULT_WEIGHT = 2800.0
DEFAULT_ACCELERATION = 15.5
DEFAULT_YEAR_INDEX = 8  # 1978 (78)
DEFAULT_ORIGIN_INDEX = 0  # USA

if "cylinders" not in st.session_state:
    st.session_state.cylinders = DEFAULT_CYLINDERS
if "displacement" not in st.session_state:
    st.session_state.displacement = DEFAULT_DISPLACEMENT
if "horsepower" not in st.session_state:
    st.session_state.horsepower = DEFAULT_HORSEPOWER
if "weight" not in st.session_state:
    st.session_state.weight = DEFAULT_WEIGHT
if "acceleration" not in st.session_state:
    st.session_state.acceleration = DEFAULT_ACCELERATION
if "model_year_idx" not in st.session_state:
    st.session_state.model_year_idx = DEFAULT_YEAR_INDEX
if "origin_idx" not in st.session_state:
    st.session_state.origin_idx = DEFAULT_ORIGIN_INDEX

def reset_inputs():
    st.session_state.cylinders = DEFAULT_CYLINDERS
    st.session_state.displacement = DEFAULT_DISPLACEMENT
    st.session_state.horsepower = DEFAULT_HORSEPOWER
    st.session_state.weight = DEFAULT_WEIGHT
    st.session_state.acceleration = DEFAULT_ACCELERATION
    st.session_state.model_year_idx = DEFAULT_YEAR_INDEX
    st.session_state.origin_idx = DEFAULT_ORIGIN_INDEX

ORIGIN_MAP = {"USA": 1, "Europe": 2, "Japan": 3}
YEAR_OPTIONS = [f"19{yr} ({yr})" for yr in range(70, 83)]

# Specification Input Cards (2 Columns)
col_engine, col_vehicle = st.columns(2, gap="large")

with col_engine:
    st.markdown('<div class="spec-card"><div class="card-header">⚙️ ENGINE CONFIGURATION</div>', unsafe_allow_html=True)
    
    cylinders_val = st.selectbox(
        "Cylinders",
        options=[3, 4, 5, 6, 8],
        index=[3, 4, 5, 6, 8].index(st.session_state.cylinders)
    )
    
    displacement_val = st.number_input(
        "Displacement (cu. in.)",
        min_value=50.0,
        max_value=500.0,
        value=float(st.session_state.displacement),
        step=5.0
    )
    
    horsepower_val = st.number_input(
        "Horsepower (hp)",
        min_value=40.0,
        max_value=350.0,
        value=float(st.session_state.horsepower),
        step=5.0
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

with col_vehicle:
    st.markdown('<div class="spec-card"><div class="card-header">🚗 VEHICLE DYNAMICS</div>', unsafe_allow_html=True)
    
    weight_val = st.number_input(
        "Weight (lbs)",
        min_value=1500.0,
        max_value=6000.0,
        value=float(st.session_state.weight),
        step=50.0
    )
    
    acceleration_val = st.number_input(
        "Acceleration (0-60 mph sec)",
        min_value=5.0,
        max_value=30.0,
        value=float(st.session_state.acceleration),
        step=0.5
    )
    
    year_display = st.selectbox(
        "Model Year",
        options=YEAR_OPTIONS,
        index=st.session_state.model_year_idx
    )
    model_year_val = int(year_display.split("(")[1].replace(")", ""))
    
    origin_choice = st.selectbox(
        "Origin",
        options=list(ORIGIN_MAP.keys()),
        index=st.session_state.origin_idx
    )
    origin_val = ORIGIN_MAP[origin_choice]
    
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Action Buttons
col_btn_predict, col_btn_reset = st.columns([3, 1], gap="medium")

with col_btn_predict:
    predict_clicked = st.button("🚀 PREDICT MPG", type="primary", use_container_width=True)

with col_btn_reset:
    reset_clicked = st.button("🔄 RESET", type="secondary", use_container_width=True, on_click=reset_inputs)

# ==============================================================================
# 6. PREDICTION LOGIC & ANIMATED SPEEDOMETER RESULT
# ==============================================================================
if predict_clicked:
    if displacement_val <= 0 or horsepower_val <= 0 or weight_val <= 0 or acceleration_val <= 0:
        st.error("⚠️ Parameters must be positive non-zero values.")
    else:
        # Animated Speedometer Calculation State
        calc_placeholder = st.empty()
        calc_html = f'<div class="result-card-container calc-card"><div class="result-header-label">SYSTEM DIAGNOSTIC • CALCULATING MPG</div><div class="gauge-wrapper"><svg viewBox="0 0 200 120" class="gauge-svg"><defs><linearGradient id="calc-grad" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#00F0FF"/><stop offset="100%" stop-color="#3B82F6"/></linearGradient></defs><path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="rgba(56, 189, 248, 0.15)" stroke-width="12" stroke-linecap="round"/><path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="url(#calc-grad)" stroke-width="12" stroke-linecap="round" stroke-dasharray="251.32" stroke-dashoffset="100" class="calc-arc-rev"/><text x="14" y="105" class="speedo-tick-text">0</text><text x="32" y="38" class="speedo-tick-text">15</text><text x="100" y="8" class="speedo-tick-text">30</text><text x="168" y="38" class="speedo-tick-text">45</text><text x="186" y="105" class="speedo-tick-text">60</text><g class="calc-needle-rev"><line x1="100" y1="100" x2="100" y2="30" stroke="#00F0FF" stroke-width="3.5" stroke-linecap="round" filter="drop-shadow(0 0 8px #00F0FF)"/><circle cx="100" cy="100" r="7" fill="#00F0FF" stroke="#060913" stroke-width="2"/></g></svg><div class="gauge-value-overlay"><div class="gauge-number calc-pulse-text">--.-</div><div class="gauge-unit-text">ANALYZING ENGINE</div></div></div></div>'
        calc_placeholder.markdown(calc_html, unsafe_allow_html=True)
        
        # Model Prediction Calculation
        features = pd.DataFrame([{
            "cylinders": float(cylinders_val),
            "displacement": float(displacement_val),
            "horsepower": float(horsepower_val),
            "weight": float(weight_val),
            "acceleration": float(acceleration_val),
            "model_year": float(model_year_val),
            "origin": float(origin_val)
        }])
        
        prediction = model.predict(features)
        predicted_mpg = float(prediction[0])
        
        time.sleep(0.75)  # Pause to show live diagnostic revving speedometer
        calc_placeholder.empty()
        
        save_prediction(
            cylinders_val, displacement_val, horsepower_val, weight_val, acceleration_val, model_year_val, origin_choice, predicted_mpg
        )
        
        # Speedometer Gauge Arc (Max 60 MPG) & Physical Needle Angle (-90 deg to +90 deg)
        gauge_pct = min(max(predicted_mpg / 60.0, 0.0), 1.0)
        dash_offset = 251.32 * (1.0 - gauge_pct)
        needle_angle = -90.0 + (gauge_pct * 180.0)
        
        if predicted_mpg >= 30.0:
            badge_class = "badge-high-eff"
            badge_text = "🌱 HIGH EFFICIENCY"
            grad_color = "#34D399"
        elif predicted_mpg >= 20.0:
            badge_class = "badge-good-eff"
            badge_text = "⚡ GOOD EFFICIENCY"
            grad_color = "#00F0FF"
        else:
            badge_class = "badge-low-eff"
            badge_text = "⛽ LOW EFFICIENCY"
            grad_color = "#F87171"
        
        result_html = f'<div class="result-card-container"><div class="result-header-label">ESTIMATED FUEL EFFICIENCY</div><div class="gauge-wrapper"><svg viewBox="0 0 200 120" class="gauge-svg"><defs><linearGradient id="gauge-grad" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#00F0FF"/><stop offset="100%" stop-color="{grad_color}"/></linearGradient></defs><path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="rgba(56, 189, 248, 0.15)" stroke-width="12" stroke-linecap="round"/><path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="url(#gauge-grad)" stroke-width="12" stroke-linecap="round" stroke-dasharray="251.32" stroke-dashoffset="{dash_offset:.2f}" class="gauge-meter-path"/><text x="14" y="105" class="speedo-tick-text">0</text><text x="32" y="38" class="speedo-tick-text">15</text><text x="100" y="8" class="speedo-tick-text">30</text><text x="168" y="38" class="speedo-tick-text">45</text><text x="186" y="105" class="speedo-tick-text">60</text><g class="speedo-needle-group" style="transform: rotate({needle_angle:.1f}deg); transform-origin: 100px 100px;"><line x1="100" y1="100" x2="100" y2="30" stroke="#00F0FF" stroke-width="3.5" stroke-linecap="round" filter="drop-shadow(0 0 6px #00F0FF)"/><circle cx="100" cy="100" r="7" fill="#00F0FF" stroke="#060913" stroke-width="2"/></g></svg><div class="gauge-value-overlay"><div class="gauge-number">{predicted_mpg:.1f}</div><div class="gauge-unit-text">MPG</div></div></div><div><span class="efficiency-badge {badge_class}">{badge_text}</span></div><div class="profile-strip"><div class="profile-pill"><span class="icon">⚙️</span> {cylinders_val} CYL</div><div class="profile-pill"><span class="icon">📐</span> {displacement_val:.0f} CID</div><div class="profile-pill"><span class="icon">⚡</span> {horsepower_val:.0f} HP</div><div class="profile-pill"><span class="icon">⚖️</span> {weight_val:.0f} LBS</div><div class="profile-pill"><span class="icon">⏱️</span> {acceleration_val:.1f} SEC</div><div class="profile-pill"><span class="icon">📅</span> 19{model_year_val}</div><div class="profile-pill"><span class="icon">🌍</span> {origin_choice.upper()}</div></div></div>'
        st.markdown(result_html, unsafe_allow_html=True)

# ==============================================================================
# 7. SYSTEM LOGS & HISTORY (MINIMAL COLLAPSIBLE DRAWER)
# ==============================================================================
st.markdown("<br><hr style='border-color: rgba(0, 240, 255, 0.1);'><br>", unsafe_allow_html=True)

with st.expander("⚙️ SYSTEM LOGS / PREDICTION HISTORY", expanded=False):
    history_df = get_prediction_history()
    if history_df.empty:
        st.info("No prediction logs recorded yet.")
    else:
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear History Data", type="secondary"):
            clear_prediction_history()
            st.rerun()
