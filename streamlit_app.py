# ============================================================================
# TOUCHLESS SATISFACTION SURVEY SYSTEM
# Using Teachable Machine Shareable Link (No Download Needed!)
# ============================================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import requests
import json
from io import BytesIO

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Touchless Satisfaction Survey",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# TEACHABLE MACHINE MODEL URL (Get this from "Export Model" → Copy Link)
# Example: https://teachablemachine.withgoogle.com/models/YOUR_MODEL_ID/
MODEL_URL = "https://teachablemachine.withgoogle.com/models/w3jVYBYFB/"

# Survey questions (customize these)
SURVEY_QUESTIONS = [
    "How satisfied are you with the workshop content?",
    "How satisfied are you with the instructor's teaching?",
    "How satisfied are you with the workshop materials?",
    "How satisfied are you with the hands-on activities?",
    "How satisfied are you with the overall workshop experience?"
]

# Gesture to response mapping
GESTURE_MAP = {
    'thumbs_up': {'label': 'Satisfied', 'score': 4, 'emoji': '👍'},
    'heart_sign': {'label': 'Very Satisfied', 'score': 5, 'emoji': '❤️'},
    'thumbs_down': {'label': 'Unsatisfied', 'score': 2, 'emoji': '👎'},
    'waving_finger': {'label': 'Very Unsatisfied', 'score': 1, 'emoji': '☝️'},
    'closed_fist': {'label': 'No Answer', 'score': None, 'emoji': '✊'}
}

# Google Sheets name
SHEET_NAME = "Survey_Responses"

# ============================================================================
# LOAD MODEL FROM TEACHABLE MACHINE LINK
# ============================================================================

@st.cache_resource
def load_model_from_url(model_url):
    """Load Teachable Machine model from shareable link"""
    try:
        # Ensure URL ends with /
        if not model_url.endswith('/'):
            model_url += '/'
        
        # Load model.json to get class names
        metadata_url = model_url + 'metadata.json'
        response = requests.get(metadata_url)
        metadata = response.json()
        
        # Extract class names
        class_names = [label['className'] for label in metadata['labels']]
        
        # Load the model
        model_json_url = model_url + 'model.json'
        model = tf.keras.models.model_from_json(
            requests.get(model_json_url).text
        )
        
        # Load weights
        weights_url = model_url + 'weights.bin'
        weights_response = requests.get(weights_url)
        weights_buffer = BytesIO(weights_response.content)
        model.load_weights(weights_buffer)
        
        return model, class_names
    
    except Exception as e:
        st.error(f"Error loading model from URL: {e}")
        st.info("Please check your Teachable Machine model URL")
        return None, None

# ============================================================================
# GOOGLE SHEETS CONNECTION (FIXED)
# ============================================================================

def connect_to_sheets():
    """Connect to Google Sheets using Streamlit secrets"""
    try:
        # Define scope
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        
        # Load credentials from Streamlit secrets
        credentials_dict = dict(st.secrets["google_credentials"])
        credentials = Credentials.from_service_account_info(
            credentials_dict,
            scopes=scope
        )
        
        # Authorize client
        client = gspread.authorize(credentials)
        
        # Try to open existing sheet, create if doesn't exist
        try:
            sheet = client.open(SHEET_NAME).sheet1
            st.success(f"Connected to existing sheet: {SHEET_NAME}")
        except gspread.exceptions.SpreadsheetNotFound:
            # Create new spreadsheet
            spreadsheet = client.create(SHEET_NAME)
            sheet = spreadsheet.sheet1
            
            # Add headers
            headers = [
                'Timestamp', 'Respondent_Name', 'Organization',
                'Q1_Label', 'Q1_Score', 'Q1_Confidence',
                'Q2_Label', 'Q2_Score', 'Q2_Confidence',
                'Q3_Label', 'Q3_Score', 'Q3_Confidence',
                'Q4_Label', 'Q4_Score', 'Q4_Confidence',
                'Q5_Label', 'Q5_Score', 'Q5_Confidence'
            ]
            sheet.append_row(headers)
            
            # Share with everyone (optional - or share manually)
            spreadsheet.share('', perm_type='anyone', role='writer')
            
            st.success(f"Created new sheet: {SHEET_NAME}")
        
        return sheet
    
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        st.info("💡 Make sure you've added Google credentials to Streamlit secrets")
        return None

def save_to_sheets(sheet, data):
    """Save survey response to Google Sheets"""
    try:
        sheet.append_row(data)
        return True
    except Exception as e:
        st.error(f"Error saving to sheets: {e}")
        return False

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize to 224x224 (Teachable Machine standard)
    img = image.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_gesture(model, class_names, image):
    """Predict hand gesture from image"""
    try:
        # Preprocess
        img_array = preprocess_image(image)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        probabilities = predictions[0]
        
        # Get predicted class
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        predicted_gesture = class_names[predicted_idx]
        
        return predicted_gesture, confidence, probabilities
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0, []

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("✋ Touchless Satisfaction Survey")
    st.markdown("### Use hand gestures to answer survey questions!")
    
    # Sidebar - Instructions & Configuration
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        **How to respond:**
        
        ❤️ **Heart Sign** = Very Satisfied (5)
        
        👍 **Thumbs Up** = Satisfied (4)
        
        👎 **Thumbs Down** = Unsatisfied (2)
        
        ☝️ **Waving Finger** = Very Unsatisfied (1)
        
        ✊ **Closed Fist** = No Answer
        
        ---
        
        **Steps:**
        1. Read the question
        2. Click "Take photo"
        3. Show your hand gesture
        4. Capture response
        5. Confirm and move to next question
        """)
        
        st.markdown("---")
        
        # Model URL input (for easy switching)
        st.subheader("⚙️ Configuration")
        model_url_input = st.text_input(
            "Teachable Machine Model URL:",
            value=MODEL_URL,
            help="Get this from Teachable Machine → Export → Copy Link"
        )
        
        st.info("💡 Make sure your hand is clearly visible!")
    
    # Initialize session state
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
        st.session_state.responses = []
        st.session_state.survey_started = False
        st.session_state.survey_completed = False
        st.session_state.respondent_name = ""
        st.session_state.model_url = model_url_input
    
    # Update model URL if changed
    if model_url_input != st.session_state.model_url:
        st.session_state.model_url = model_url_input
        st.cache_resource.clear()
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, class_names = load_model_from_url(st.session_state.model_url)
    
    if model is None or class_names is None:
        st.error("❌ Failed to load model. Please check the URL and try again.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Survey start screen
    if not st.session_state.survey_started:
        st.markdown("## Welcome to the Survey!")
        st.markdown("Please enter your information to begin:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Your Name (Optional):", key="name_input")
        
        with col2:
            organization = st.text_input("Organization/Institution:", key="org_input")
        
        if st.button("🚀 Start Survey", type="primary", use_container_width=True):
            st.session_state.respondent_name = name if name else "Anonymous"
            st.session_state.organization = organization if organization else "N/A"
            st.session_state.survey_started = True
            st.session_state.start_time = datetime.now()
            st.rerun()
        
        return
    
    # Survey completed screen
    if st.session_state.survey_completed:
        st.success("✅ Survey Completed! Thank you for your feedback!")
        
        # Display summary
        st.markdown("## Your Responses Summary")
        
        summary_data = []
        for i, response in enumerate(st.session_state.responses):
            summary_data.append({
                'Question': f"Q{i+1}",
                'Response': response['label'],
                'Score': response['score'] if response['score'] else 'N/A',
                'Confidence': f"{response['confidence']:.1%}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Calculate average score
        scores = [r['score'] for r in st.session_state.responses if r['score'] is not None]
        if scores:
            avg_score = sum(scores) / len(scores)
            st.metric("Average Satisfaction Score", f"{avg_score:.2f} / 5.0")
        
        if st.button("📝 Submit Another Response"):
            # Reset session
            st.session_state.current_question = 0
            st.session_state.responses = []
            st.session_state.survey_started = False
            st.session_state.survey_completed = False
            st.rerun()
        
        return
    
    # Survey in progress
    current_q = st.session_state.current_question
    total_q = len(SURVEY_QUESTIONS)
    
    # Progress bar
    progress = (current_q / total_q)
    st.progress(progress, text=f"Question {current_q + 1} of {total_q}")
    
    # Display current question
    st.markdown(f"## Question {current_q + 1}")
    st.markdown(f"### {SURVEY_QUESTIONS[current_q]}")
    
    # Camera input
    st.markdown("---")
    st.markdown("**Show your hand gesture and capture:**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Camera input
        img_file = st.camera_input("Camera", key=f"camera_{current_q}")
        
        if img_file is not None:
            # Load image
            image = Image.open(img_file)
            
            # Predict gesture
            with st.spinner("Analyzing gesture..."):
                gesture, confidence, probs = predict_gesture(model, class_names, image)
            
            if gesture:
                gesture_info = GESTURE_MAP.get(gesture, {
                    'label': gesture,
                    'score': 3,
                    'emoji': '🤷'
                })
                
                # Display result
                st.success(f"Detected: {gesture_info['emoji']} {gesture_info['label']}")
                st.info(f"Confidence: {confidence:.1%}")
                
                # Show all predictions
                with st.expander("View all predictions"):
                    for i, (class_name, prob) in enumerate(zip(class_names, probs)):
                        st.write(f"{class_name}: {prob:.1%}")
                
                # Confirm button
                if st.button("✅ Confirm This Response", type="primary", use_container_width=True):
                    # Save response
                    response_data = {
                        'question': SURVEY_QUESTIONS[current_q],
                        'gesture': gesture,
                        'label': gesture_info['label'],
                        'score': gesture_info['score'],
                        'confidence': confidence,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    st.session_state.responses.append(response_data)
                    
                    # Move to next question or complete
                    if current_q < total_q - 1:
                        st.session_state.current_question += 1
                        st.rerun()
                    else:
                        # Survey completed - save to Google Sheets
                        st.session_state.survey_completed = True
                        
                        # Prepare data for sheets
                        with st.spinner("Saving to Google Sheets..."):
                            sheet = connect_to_sheets()
                            if sheet:
                                row_data = [
                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    st.session_state.respondent_name,
                                    st.session_state.organization
                                ]
                                
                                # Add all responses
                                for resp in st.session_state.responses:
                                    row_data.extend([
                                        resp['label'],
                                        resp['score'] if resp['score'] else 'N/A',
                                        f"{resp['confidence']:.2%}"
                                    ])
                                
                                if save_to_sheets(sheet, row_data):
                                    st.success("✅ Saved to Google Sheets!")
                                else:
                                    st.warning("⚠️ Could not save to Google Sheets")
                        
                        st.rerun()
    
    with col2:
        st.markdown("**Gesture Guide:**")
        for gesture, info in GESTURE_MAP.items():
            st.markdown(f"{info['emoji']} {info['label']}")
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if current_q > 0:
            if st.button("⬅️ Previous Question"):
                st.session_state.current_question -= 1
                st.rerun()
    
    with col2:
        if st.button("🔄 Reset Survey", type="secondary"):
            st.session_state.current_question = 0
            st.session_state.responses = []
            st.session_state.survey_started = False
            st.rerun()
    
    with col3:
        if current_q < total_q - 1:
            if st.button("Skip Question ➡️"):
                # Save as no answer
                response_data = {
                    'question': SURVEY_QUESTIONS[current_q],
                    'gesture': 'closed_fist',
                    'label': 'No Answer',
                    'score': None,
                    'confidence': 1.0,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                st.session_state.responses.append(response_data)
                st.session_state.current_question += 1
                st.rerun()

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
