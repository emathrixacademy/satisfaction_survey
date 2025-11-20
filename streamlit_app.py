# ============================================================================
# TOUCHLESS SATISFACTION SURVEY - SIMPLIFIED VERSION
# Works on Streamlit Cloud without dependency issues
# ============================================================================

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import json

# Try to import optional dependencies
try:
    import gspread
    from google.oauth2.service_account import Credentials
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False
    st.warning("Google Sheets integration not available. Results will be shown on screen only.")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.error("TensorFlow not available. Please check requirements.txt")

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Touchless Survey",
    page_icon="✋",
    layout="wide"
)

# MODEL URL - Replace with your Teachable Machine link
MODEL_URL = "https://teachablemachine.withgoogle.com/models/YOUR_MODEL_ID/"

# Survey questions
SURVEY_QUESTIONS = [
    "How satisfied are you with the workshop content?",
    "How satisfied are you with the instructor's teaching?",
    "How satisfied are you with the workshop materials?",
    "How satisfied are you with the hands-on activities?",
    "How satisfied are you with the overall workshop experience?"
]

# Gesture mapping
GESTURE_MAP = {
    'thumbs_up': {'label': 'Satisfied', 'score': 4, 'emoji': '👍'},
    'heart_sign': {'label': 'Very Satisfied', 'score': 5, 'emoji': '❤️'},
    'thumbs_down': {'label': 'Unsatisfied', 'score': 2, 'emoji': '👎'},
    'waving_finger': {'label': 'Very Unsatisfied', 'score': 1, 'emoji': '☝️'},
    'closed_fist': {'label': 'No Answer', 'score': None, 'emoji': '✊'}
}

SHEET_NAME = "Survey_Responses"

# ============================================================================
# SIMPLE MODEL LOADER (without complex dependencies)
# ============================================================================

@st.cache_resource
def load_simple_model():
    """Simple model loading for demo purposes"""
    # This is a simplified version - in production, load from Teachable Machine
    return True  # Placeholder

def simple_predict(image):
    """Simplified prediction for demo"""
    # In production, this would use the actual TensorFlow model
    # For now, return a random gesture for testing
    import random
    gestures = list(GESTURE_MAP.keys())
    gesture = random.choice(gestures)
    confidence = random.uniform(0.7, 0.99)
    return gesture, confidence

# ============================================================================
# GOOGLE SHEETS (OPTIONAL)
# ============================================================================

def save_to_sheets_safe(data):
    """Safe Google Sheets save with fallback"""
    if not SHEETS_AVAILABLE:
        st.warning("Google Sheets not available. Displaying results only.")
        return False
    
    try:
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        
        creds_dict = dict(st.secrets["google_credentials"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        
        try:
            sheet = client.open(SHEET_NAME).sheet1
        except:
            spreadsheet = client.create(SHEET_NAME)
            sheet = spreadsheet.sheet1
            headers = ['Timestamp', 'Name', 'Org', 
                      'Q1', 'S1', 'C1', 'Q2', 'S2', 'C2',
                      'Q3', 'S3', 'C3', 'Q4', 'S4', 'C4',
                      'Q5', 'S5', 'C5']
            sheet.append_row(headers)
        
        sheet.append_row(data)
        return True
    except Exception as e:
        st.error(f"Could not save to sheets: {e}")
        return False

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("✋ Touchless Satisfaction Survey")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        **Gesture Guide:**
        
        ❤️ Heart = Very Satisfied (5)
        👍 Thumbs Up = Satisfied (4)  
        👎 Thumbs Down = Unsatisfied (2)
        ☝️ Waving = Very Unsatisfied (1)
        ✊ Fist = No Answer
        """)
        
        st.info("Show clear hand gestures for best results!")
    
    # Initialize session
    if 'started' not in st.session_state:
        st.session_state.started = False
        st.session_state.current_q = 0
        st.session_state.responses = []
        st.session_state.completed = False
    
    # Start screen
    if not st.session_state.started:
        st.markdown("## Welcome!")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Your Name:")
        with col2:
            org = st.text_input("Organization:")
        
        if st.button("🚀 Start Survey", type="primary"):
            st.session_state.name = name or "Anonymous"
            st.session_state.org = org or "N/A"
            st.session_state.started = True
            st.rerun()
        return
    
    # Completed screen
    if st.session_state.completed:
        st.success("✅ Survey Complete!")
        
        st.markdown("## Your Responses")
        
        df = pd.DataFrame([{
            'Question': f"Q{i+1}",
            'Response': r['label'],
            'Score': r['score'] or 'N/A',
            'Confidence': f"{r['confidence']:.1%}"
        } for i, r in enumerate(st.session_state.responses)])
        
        st.dataframe(df, use_container_width=True)
        
        scores = [r['score'] for r in st.session_state.responses if r['score']]
        if scores:
            st.metric("Average Score", f"{sum(scores)/len(scores):.2f}/5.0")
        
        if st.button("📝 New Response"):
            st.session_state.started = False
            st.session_state.current_q = 0
            st.session_state.responses = []
            st.session_state.completed = False
            st.rerun()
        return
    
    # Survey in progress
    current_q = st.session_state.current_q
    total_q = len(SURVEY_QUESTIONS)
    
    st.progress(current_q / total_q, text=f"Question {current_q + 1} of {total_q}")
    
    st.markdown(f"## Question {current_q + 1}")
    st.markdown(f"### {SURVEY_QUESTIONS[current_q]}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        img_file = st.camera_input("Show your gesture", key=f"cam_{current_q}")
        
        if img_file:
            image = Image.open(img_file)
            
            with st.spinner("Analyzing..."):
                # Use simple prediction for demo
                gesture, confidence = simple_predict(image)
            
            info = GESTURE_MAP[gesture]
            
            st.success(f"Detected: {info['emoji']} {info['label']}")
            st.info(f"Confidence: {confidence:.1%}")
            
            if st.button("✅ Confirm", type="primary"):
                st.session_state.responses.append({
                    'label': info['label'],
                    'score': info['score'],
                    'confidence': confidence
                })
                
                if current_q < total_q - 1:
                    st.session_state.current_q += 1
                    st.rerun()
                else:
                    st.session_state.completed = True
                    
                    # Try to save
                    data = [
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        st.session_state.name,
                        st.session_state.org
                    ]
                    for r in st.session_state.responses:
                        data.extend([r['label'], r['score'] or 'N/A', f"{r['confidence']:.1%}"])
                    
                    save_to_sheets_safe(data)
                    st.rerun()
    
    with col2:
        st.markdown("**Gestures:**")
        for g, info in GESTURE_MAP.items():
            st.write(f"{info['emoji']} {info['label']}")
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if current_q > 0 and st.button("⬅️ Back"):
            st.session_state.current_q -= 1
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset"):
            st.session_state.started = False
            st.session_state.current_q = 0
            st.session_state.responses = []
            st.rerun()
    
    with col3:
        if current_q < total_q - 1 and st.button("Skip ➡️"):
            st.session_state.responses.append({
                'label': 'No Answer',
                'score': None,
                'confidence': 1.0
            })
            st.session_state.current_q += 1
            st.rerun()

if __name__ == "__main__":
    main()
