import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
import tempfile
import tf_keras
import time
import random
import json
import re

# ==========================================
# 1. PAGE CONFIGURATION & INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="AI Exercise Classifier & Guide",
    page_icon="🏋️‍♂️",
    layout="wide"
)

# Initialize Session States
if "frames_analyzed" not in st.session_state:
    st.session_state.frames_analyzed = 0
if "detections_today" not in st.session_state:
    st.session_state.detections_today = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "last_detected_time" not in st.session_state:
    st.session_state.last_detected_time = "--:--:--"
if "coach_guidance" not in st.session_state:
    st.session_state.coach_guidance = None
if "last_coach_exercise" not in st.session_state:
    st.session_state.last_coach_exercise = None
if "detected_exercise" not in st.session_state:
    st.session_state.detected_exercise = None
if "confidence_value" not in st.session_state:
    st.session_state.confidence_value = 0.0
if "prediction_scores" not in st.session_state:
    st.session_state.prediction_scores = np.array([0.0, 0.0, 0.0])

# ==========================================
# 2. PREMIUM fitness-dashboard THEME (CUSTOM CSS)
# ==========================================
dashboard_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

/* Entire App Background & Fonts */
.stApp {
    background-color: #2C2F33;
    color: #E6E6E6;
    font-family: 'Inter', sans-serif;
}

/* Hide Default Streamlit Footer */
footer {
    visibility: hidden;
}

/* Main Layout Padding */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif;
    color: #E6E6E6 !important;
    font-weight: 700;
}

/* Standard Paragraphs & Muted text */
p {
    color: #A0A4A8;
    font-size: 15px;
    line-height: 1.6;
}

/* Three Columns Styled as Cards */
div[data-testid="stColumn"], div[data-testid="column"] {
    background-color: #3A3F44 !important;
    color: #E6E6E6 !important;
    padding: 24px !important;
    border-radius: 20px !important;
    border: 1px solid #4A4F55 !important;
    box-shadow: 0px 8px 30px rgba(0, 0, 0, 0.2) !important;
    margin-bottom: 20px !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}

div[data-testid="stColumn"]:hover, div[data-testid="column"]:hover {
    border-color: #B5FF00 !important;
    box-shadow: 0px 10px 30px rgba(181, 255, 0, 0.1) !important;
}

/* Streamlit Selectbox/Dropdown styles */
div[data-baseweb="select"] > div {
    background-color: #2C2F33 !important;
    border: 1px solid #4A4F55 !important;
    color: #E6E6E6 !important;
    border-radius: 8px !important;
    padding: 2px 4px !important;
}

/* Streamlit File Uploader Override */
section[data-testid="stFileUploader"] {
    background-color: #3A3F44 !important;
    border: 2px dashed #4A4F55 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    transition: border-color 0.25s ease;
}
section[data-testid="stFileUploader"]:hover {
    border-color: #B5FF00 !important;
}

/* Custom Buttons Styling */
.stButton > button {
    width: 100%;
    background-color: #B5FF00 !important;
    color: #2C2F33 !important;
    border: 1px solid #B5FF00 !important;
    border-radius: 30px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background-color: #8DFF3F !important;
    border-color: #8DFF3F !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(181, 255, 0, 0.2) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Red Stop Button Override for Webcam stream */
.stButton > button[class*="stop-btn"] {
    background-color: #FF4D4D !important;
    border-color: #FF4D4D !important;
    color: #FFFFFF !important;
}

.stButton > button[class*="stop-btn"]:hover {
    background-color: #E60000 !important;
    border-color: #E60000 !important;
    box-shadow: 0 4px 12px rgba(255, 77, 77, 0.25) !important;
}

/* Toast custom overlay overrides */
div[data-testid="stToast"] {
    background-color: #3A3F44 !important;
    border: 1px solid #B5FF00 !important;
    border-radius: 8px !important;
    color: #E6E6E6 !important;
}

/* Scrollbar customization */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #2C2F33;
}
::-webkit-scrollbar-thumb {
    background: #4A4F55;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #6B7075;
}

/* Streamlit Native Tabs Styling */
button[data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #A0A4A8 !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.25s ease !important;
    background-color: transparent !important;
}
button[data-baseweb="tab"]:hover {
    color: #B5FF00 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #B5FF00 !important;
    border-bottom: 2px solid #B5FF00 !important;
}
</style>
"""

st.markdown(dashboard_style, unsafe_allow_html=True)

# Title & Dashboard Header
st.markdown("""
<div style="text-align: center; margin-bottom: 25px;">
    <h1 style="font-size: 2.5rem; margin-bottom: 4px; color: #E6E6E6 !important;">
        🏋️‍♂️ AI WORKOUT MONITOR
    </h1>
    <p style="color: #A0A4A8; font-size: 1.05rem; margin: 0;">
        Real-time telemetry tracking & movement coaching guide.
    </p>
</div>
""", unsafe_allow_html=True)

CLASS_NAMES = ["Pull-Ups", "Push-Ups", "Squats"]
CLASS_ICONS = {
    "Pull-Ups": "🤸",
    "Push-Ups": "💪",
    "Squats": "🦵"
}

# ==========================================
# 3. MODEL LOADING (Keras)
# ==========================================
@st.cache_resource
def load_my_model():
    import tf_keras as tfk
    model = tfk.models.load_model('keras_model.h5', compile=False)
    return model

model = None
try:
    model = load_my_model()
    if "model_toast_shown" not in st.session_state:
        st.toast("Core neural model loaded successfully!", icon="✅")
        st.session_state.model_toast_shown = True
except Exception as e:
    st.error(f"Error loading keras neural model: {e}")
    st.stop()

@st.cache_data
def load_muscle_workouts():
    try:
        with open("muscle_workouts.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading muscle workouts: {e}")
        return {}

@st.cache_data
def load_exercise_guide():
    try:
        with open("exercise_guide.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading exercise guide: {e}")
        return {}

def render_exercise_card(ex_name, ex_info):
    gif_url = (ex_info.get("gif_url") or "").strip()
    image_url = (ex_info.get("image_url") or "").strip()
    steps = ex_info.get("steps", [])
    
    if gif_url:
        media_url = gif_url
        badge_text = "GIF"
    elif image_url:
        media_url = image_url
        badge_text = "IMG"
    else:
        media_url = None
        badge_text = None
    
    if media_url:
        badge_html = f'''<div style="position: absolute; top: 6px; right: 6px; font-size: 9px; padding: 2px 6px; border-radius: 4px; background-color: rgba(0, 0, 0, 0.65); color: #B5FF00; font-weight: 700; letter-spacing: 0.5px; z-index: 10; pointer-events: none;">{badge_text}</div>'''
        left_box = f'''<div style="position: relative; width: 100%; height: 160px; overflow: hidden; border-radius: 8px;">
            {badge_html}
            <img src="{media_url}" alt="{ex_name}" style="width: 100%; height: 160px; object-fit: cover; border-radius: 8px; display: block;" />
        </div>'''
    else:
        left_box = '''<div style="width: 100%; height: 160px; background-color: #3A3F44; border: 2px dashed #4A4F55; border-radius: 8px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #6B7075; text-align: center; padding: 10px; box-sizing: border-box;">
            <span style="font-size: 28px; margin-bottom: 4px; filter: grayscale(100%); opacity: 0.6;">🖼️</span>
            <span style="font-size: 11px; font-weight: 600; color: #6B7075; text-transform: uppercase; letter-spacing: 0.5px;">No preview available</span>
        </div>'''
        
    steps_html = "".join([
        f'<li style="margin-bottom: 6px; color: #E6E6E6; font-size: 13px; line-height: 1.4;"><span style="color: #B5FF00; font-weight: 700; font-size: 14px; margin-right: 4px;">{i+1}.</span> {step}</li>'
        for i, step in enumerate(steps)
    ])
    
    if not steps_html:
        steps_html = '<li style="color: #A0A4A8; font-size: 13px;">Follow standard movement guidelines.</li>'
        
    card_html = f'''
    <div style="max-width: 540px; background-color: #2C2F33; border: 1px solid #4A4F55; border-radius: 10px; padding: 12px; margin-top: 5px; margin-bottom: 10px;">
        <div style="display: grid; grid-template-columns: 1fr 1.3fr; gap: 10px; align-items: center;">
            <div style="width: 100%; height: 160px;">
                {left_box}
            </div>
            <div style="height: 160px; max-height: 160px; overflow-y: auto; padding-right: 6px;">
                <ol style="margin: 0; padding-left: 0; list-style-type: none;">
                    {steps_html}
                </ol>
            </div>
        </div>
    </div>
    '''
    st.markdown(card_html, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def generate_routine(muscle_group, exercises, api_key):
    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import PromptTemplate
        
        llm = ChatGroq(
            temperature=0.7, 
            groq_api_key=api_key, 
            model_name="llama-3.1-8b-instant"
        )
        
        template = """
        You are a professional fitness coach.
        The user wants to work on the following muscle group: {muscle_group}.
        
        Please build a short structured workout routine using ONLY the following list of exercises: {exercises}.
        Do not invent or add any exercises outside of this list.
        
        Provide your response in the following exact format:
        
        [WARMUP]
        <A brief warm-up note for this muscle group>
        
        [ROUTINE]
        * <Exercise 1> - <Suggested sets/reps>
        * <Exercise 2> - <Suggested sets/reps>
        * <Exercise 3> - <Suggested sets/reps>
        
        [COOLDOWN]
        <A quick cool-down tip for recovery>
        """
        
        prompt = PromptTemplate(template=template, input_variables=["muscle_group", "exercises"])
        chain = prompt | llm
        
        response = chain.invoke({"muscle_group": muscle_group, "exercises": ", ".join(exercises)})
        return response.content
    except Exception as e:
        return f"### ⚠️ AI Coach Connection Error\nCould not fetch routine from Groq. Details: {e}"

def parse_routine_response(text):
    warmup = ""
    routine = []
    cooldown = ""
    
    if not text:
        return warmup, routine, cooldown
        
    parts = text.split('[')
    for part in parts:
        if part.startswith('WARMUP]'):
            warmup = part.replace('WARMUP]', '').strip()
        elif part.startswith('ROUTINE]'):
            content = part.replace('ROUTINE]', '').strip()
            routine = [line.strip('* -•') for line in content.split('\n') if line.strip()]
        elif part.startswith('COOLDOWN]'):
            cooldown = part.replace('COOLDOWN]', '').strip()
            
    if not warmup:
        warmup = "Spend 5 minutes doing light mobility to warm up your muscles."
    if not routine:
        routine = ["Perform core movements cleanly."]
    if not cooldown:
        cooldown = "Lower your heart rate with gentle breathing and static stretches."
        
    return warmup, routine, cooldown

@st.cache_data
def load_rehab_exercises():
    try:
        with open("rehab_exercises.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading rehab exercises: {e}")
        return {}

@st.cache_data(show_spinner=False)
def generate_rehab_guidance(area, exercises, api_key):
    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import PromptTemplate
        
        llm = ChatGroq(
            temperature=0.6, 
            groq_api_key=api_key, 
            model_name="llama-3.1-8b-instant"
        )
        
        template = """
        You are a general fitness assistant, NOT a medical professional or physiotherapist.
        The user has mild stiffness/soreness in: {area}
        
        Rules you must follow strictly:
        - Only reference these gentle exercises, do not invent others: {exercise_list}
        - Do NOT diagnose any condition.
        - Do NOT recommend anything beyond light stretching or mobility work.
        - Keep tone calm, simple, and reassuring.
        - Always end by recommending they consult a doctor or physiotherapist if pain persists beyond a few days or worsens.
        """
        
        prompt = PromptTemplate(template=template, input_variables=["area", "exercise_list"])
        chain = prompt | llm
        
        response = chain.invoke({"area": area, "exercise_list": ", ".join(exercises)})
        return response.content
    except Exception as e:
        return f"### ⚠️ AI Connection Error\nCould not fetch rehab guidelines. Details: {e}"

@st.cache_data
def load_protein_sources():
    try:
        with open("protein_sources.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading protein sources: {e}")
        return {"vegetarian": [], "non_vegetarian": []}

@st.cache_data(show_spinner=False)
def generate_meal_plan(goal, calorie_target, protein_target, diet_pref, protein_sources_context, api_key):
    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import PromptTemplate
        
        llm = ChatGroq(
            temperature=0.7, 
            groq_api_key=api_key, 
            model_name="llama-3.1-8b-instant"
        )
        
        template = """
        You are a professional fitness coach and nutritionist.
        Create a 1-day sample meal plan (breakfast, lunch, dinner, 1-2 snacks) for a user with the following goals and needs:
        - Goal: {goal}
        - Daily Calorie Target: ~{calorie_target} kcal
        - Daily Protein Target: ~{protein_target} g
        - Diet Preference: {diet_pref}

        Rules:
        - ONLY include foods suitable for a {diet_pref} diet.
        - Primary protein sources available to pull from: {protein_sources_context}. You may include reasonable common carbs (rice, oats, bread, sweet potatoes), healthy fats (nuts, olive oil, avocado), and vegetables to round out balanced meals.
        - Output the meal plan strictly in the following structured format:

        [BREAKFAST]
        * Item 1 - Serving - Approx Calories & Protein
        * Item 2 - Serving - Approx Calories & Protein

        [LUNCH]
        * Item 1 - Serving - Approx Calories & Protein
        * Item 2 - Serving - Approx Calories & Protein

        [SNACKS]
        * Item 1 - Serving - Approx Calories & Protein
        * Item 2 - Serving - Approx Calories & Protein

        [DINNER]
        * Item 1 - Serving - Approx Calories & Protein
        * Item 2 - Serving - Approx Calories & Protein

        [SUMMARY]
        Summary line of total estimated daily calories and protein.

        [DISCLAIMER]
        This meal plan is a general estimate based on standard nutritional formulas and is not personalized medical or dietary advice. Consult a nutritionist or doctor for specific medical conditions or dietary requirements.
        """
        
        prompt = PromptTemplate(
            template=template, 
            input_variables=["goal", "calorie_target", "protein_target", "diet_pref", "protein_sources_context"]
        )
        chain = prompt | llm
        
        response = chain.invoke({
            "goal": goal,
            "calorie_target": calorie_target,
            "protein_target": protein_target,
            "diet_pref": diet_pref,
            "protein_sources_context": protein_sources_context
        })
        return response.content
    except Exception as e:
        return f"### ⚠️ AI Meal Plan Connection Error\nCould not fetch meal plan from Groq. Details: {e}"

def parse_meal_plan_response(text):
    breakfast, lunch, snacks, dinner, summary, disclaimer = [], [], [], [], "", ""
    if not text:
        return breakfast, lunch, snacks, dinner, summary, disclaimer
        
    parts = text.split('[')
    for part in parts:
        if part.startswith('BREAKFAST]'):
            content = part.replace('BREAKFAST]', '').strip()
            breakfast = [line.strip('* -•') for line in content.split('\n') if line.strip()]
        elif part.startswith('LUNCH]'):
            content = part.replace('LUNCH]', '').strip()
            lunch = [line.strip('* -•') for line in content.split('\n') if line.strip()]
        elif part.startswith('SNACKS]'):
            content = part.replace('SNACKS]', '').strip()
            snacks = [line.strip('* -•') for line in content.split('\n') if line.strip()]
        elif part.startswith('DINNER]'):
            content = part.replace('DINNER]', '').strip()
            dinner = [line.strip('* -•') for line in content.split('\n') if line.strip()]
        elif part.startswith('SUMMARY]'):
            summary = part.replace('SUMMARY]', '').strip()
        elif part.startswith('DISCLAIMER]'):
            disclaimer = part.replace('DISCLAIMER]', '').strip()
            
    return breakfast, lunch, snacks, dinner, summary, disclaimer

# ==========================================
# 4. GROQ & LANGCHAIN COACH INTEGRATION
# ==========================================
@st.cache_data(show_spinner=False)
def get_exercise_details(exercise_name, api_key):
    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import PromptTemplate
        
        llm = ChatGroq(
            temperature=0.7, 
            groq_api_key=api_key, 
            model_name="llama-3.1-8b-instant"
        )
        
        template = """
        You are a professional fitness coach.
        The user is doing or asking about the following exercise: {exercise}
        
        Please provide the following details in simple words:
        1. What is this exercise? (Brief Description)
        2. What are its major benefits?
        - Do not include an introductory sentence before the bullet list. Start the Benefits section directly with the bullet points. Format each research benefit as a single bullet: 'Bold Title: description.'
        3. 3 Key tips for maintaining correct form and avoiding injury.
        """
        
        prompt = PromptTemplate(template=template, input_variables=["exercise"])
        chain = prompt | llm
        
        response = chain.invoke({"exercise": exercise_name})
        return response.content
    except Exception as e:
        return f"### ⚠️ AI Coach Connection Error\nCould not fetch response from Groq. Details: {e}"

# ==========================================
# 5. PREPROCESSING FUNCTION
# ==========================================
def preprocess_and_predict(image_data, model):
    target_size = (224, 224) 
    image = ImageOps.fit(image_data, target_size, Image.Resampling.LANCZOS)
    image = image.convert("RGB")
    img_array = np.asarray(image)
    normalized_img_array = img_array.astype(np.float32) / 255.0
    input_shape_img = np.expand_dims(normalized_img_array, axis=0)
    
    if hasattr(model, 'predict'):
        predictions = model.predict(input_shape_img)
    else:
        predictions = model(input_shape_img)
        
    return np.array(predictions)[0]

# ==========================================
# 6. SVG SEMI-CIRCULAR GAUGE COMPONENT
# ==========================================
def draw_svg_gauge(confidence_pct):
    r = 40
    circumference = 3.14159 * r # semi-circle path length
    dashoffset = circumference * (1.0 - (confidence_pct / 100.0))
    
    # Label Threshold Category
    if confidence_pct >= 80:
        label = "High Confidence"
        text_color = "#B5FF00"
    elif confidence_pct >= 50:
        label = "Medium Confidence"
        text_color = "#A0A4A8"
    else:
        label = "Low Confidence"
        text_color = "#FF4D4D"
        
    gauge_html = f"""
    <div style="text-align: center; margin: 10px 0;">
        <svg width="190" height="110" viewBox="0 0 100 60" style="margin: 0 auto;">
            <!-- Background Arc -->
            <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="#4A4F55" stroke-width="8" stroke-linecap="round" />
            <!-- Active Progress Arc -->
            <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="#B5FF00" 
                  stroke-width="8" stroke-linecap="round" 
                  stroke-dasharray="{circumference}" stroke-dashoffset="{dashoffset}" 
                  style="transition: stroke-dashoffset 0.4s ease-out;" />
            <!-- Metric Values -->
            <text x="50" y="44" text-anchor="middle" font-size="12" fill="#E6E6E6" font-family="'Space Grotesk', sans-serif" font-weight="bold">{confidence_pct:.1f}%</text>
            <text x="50" y="54" text-anchor="middle" font-size="5" fill="{text_color}" font-family="'Inter', sans-serif" font-weight="700" style="letter-spacing: 0.5px;">{label.upper()}</text>
        </svg>
    </div>
    """
    return gauge_html

# ==========================================
# 7. PARSE AI COACH TEXT OUTPUT
# ==========================================
def clean_bullet_line(line):
    cleaned = re.sub(r'^[-*•\d\.]+\s*', '', line).strip()
    return cleaned

def parse_coach_response(text):
    description = ""
    benefits = []
    tips = []
    
    if not text:
        return description, benefits, tips
        
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    current_section = None
    
    for line in lines:
        clean_header_lower = re.sub(r'^[\#\*\s\d\.\)]+', '', line).lower().strip()
        lower = line.lower()
        
        # Section header matching
        is_desc = "what is" in clean_header_lower or clean_header_lower.startswith("description")
        is_benefits = "benefit" in clean_header_lower
        is_tips = "tip" in clean_header_lower or "coaching" in clean_header_lower or ("form" in clean_header_lower and "tips" in clean_header_lower)
        
        if is_desc:
            current_section = "desc"
            if ":" in line:
                after = line.split(":", 1)[1].strip()
                after_clean = re.sub(r'\*+', '', after).strip()
                if after_clean and not after_clean.lower().startswith("(brief") and not after_clean.lower().endswith("?"):
                    description += " " + after_clean
            continue
            
        elif is_benefits:
            current_section = "benefits"
            if ":" in line:
                after = line.split(":", 1)[1].strip()
                after_clean = re.sub(r'\*+', '', after).strip()
                if after_clean and not after_clean.endswith(":") and "including" not in after_clean.lower() and not after_clean.lower().endswith("?"):
                    cleaned = clean_bullet_line(after_clean)
                    if cleaned and len(cleaned) > 2:
                        benefits.append(cleaned)
            continue
            
        elif is_tips:
            current_section = "tips"
            if ":" in line:
                after = line.split(":", 1)[1].strip()
                after_clean = re.sub(r'\*+', '', after).strip()
                if after_clean and not after_clean.endswith(":") and "tips" not in after_clean.lower() and not after_clean.lower().endswith("?"):
                    cleaned = clean_bullet_line(after_clean)
                    if cleaned and len(cleaned) > 2:
                        tips.append(cleaned)
            continue
            
        # Accumulate content per section
        if current_section == "desc":
            if not line.startswith('#') and not re.match(r'^\**\d+\.', line):
                description += " " + line
                
        elif current_section == "benefits":
            # Ignore intro sentences ending in colon or containing "including" if not a bullet item
            if (line.endswith(':') or "including" in line.lower() or "here are" in line.lower()) and not line.startswith(('-', '*', '•')) and not '**' in line:
                continue
            cleaned = clean_bullet_line(line)
            if cleaned and cleaned not in ["?", "**", "?**"] and len(cleaned) > 2:
                benefits.append(cleaned)
                
        elif current_section == "tips":
            if (line.endswith(':') or "here are" in line.lower()) and not line.startswith(('-', '*', '•', '1', '2', '3')) and not '**' in line:
                continue
            cleaned = clean_bullet_line(line)
            if cleaned and cleaned not in ["?", "**", "?**"] and len(cleaned) > 2:
                tips.append(cleaned)
                
    description = description.strip()
    
    # Fallback for description
    if not description:
        parts = [p.strip() for p in text.split('\n\n') if p.strip()]
        if parts:
            first_p = parts[0]
            if not any(k in first_p.lower() for k in ["benefit", "tip", "1.", "2.", "3."]):
                description = first_p

    def format_bold(item_list):
        formatted_list = []
        for item in item_list:
            fmt = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', item)
            fmt = re.sub(r'\*\*', '', fmt).strip()
            if fmt:
                formatted_list.append(fmt)
        return formatted_list

    formatted_desc = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', description)
    formatted_desc = re.sub(r'\*\*', '', formatted_desc).strip()

    return formatted_desc, format_bold(benefits)[:3], format_bold(tips)[:3]

# ==========================================
# 8. DYNAMIC ENCOURAGEMENT BOX GENERATOR
# ==========================================
def get_encouragement_message(exercise):
    encouragements = [
        "🔥 Precision in form builds unmatched strength. Control your tempo!",
        "⚡ Repetition is the mother of mastery. Focus on full range of motion!",
        "💪 Excellent effort! Ensure your core is fully engaged on every transition.",
        "🧠 Consistent posture and solid breathing are key to workout longevity.",
        "🚀 You are doing great! Power through the movement with strict form control."
    ]
    random.seed(hash(exercise))
    return random.choice(encouragements)

# ==========================================
# 9. DYNAMIC SECRETS KEY VERIFICATION
# ==========================================
groq_api_key = None
try:
    if "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

# ==========================================
# 10. TAB NAVIGATION & LAYOUT (st.tabs)
# ==========================================
tracker_tab, finder_tab, rehab_tab, calc_tab = st.tabs(["🔍 Tracker", "🎯 Workout Finder", "🩹 Rehab Guide", "🔥 Calorie & Protein Calculator"])

with tracker_tab:
    col1, col2, col3 = st.columns([1.1, 1.1, 1.2], gap="large")

    # ------------------------------------------
    # LEFT COLUMN: INPUT CARD
    # ------------------------------------------
    with col1:
        st.markdown("<h3 style='color: #E6E6E6 !important; margin-bottom: 15px;'>📸 INPUT PORTAL</h3>", unsafe_allow_html=True)
        
        input_option = st.selectbox(
            "CHOOSE TELEMETRY SOURCE:",
            ["Upload Image", "Take a Snap (Camera)", "Upload a Video File", "Live Webcam Stream"]
        )
        
        uploaded_image = None
        uploaded_video = None
        use_live_webcam = False
        
        if input_option == "Upload Image":
            uploaded_file = st.file_uploader("☁️ Drag and drop or upload image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                uploaded_image = Image.open(uploaded_file)
                
        elif input_option == "Take a Snap (Camera)":
            camera_file = st.camera_input("Pose inside target frame boundary")
            if camera_file is not None:
                uploaded_image = Image.open(camera_file)
                
        elif input_option == "Upload a Video File":
            uploaded_file = st.file_uploader("☁️ Drag and drop or upload workout video...", type=["mp4", "mov", "avi"])
            if uploaded_file is not None:
                uploaded_video = uploaded_file
            
        elif input_option == "Live Webcam Stream":
            use_live_webcam = True
            
        # --- Prediction Execution Logic ---
        detected_exercise_temp = None
        confidence_val_temp = 0.0
        prediction_scores_temp = np.array([0.0, 0.0, 0.0])
        run_prediction = False

        if uploaded_image is not None:
            st.image(uploaded_image, caption="Analyzed Image telemetry", use_container_width=True)
            prediction_scores_temp = preprocess_and_predict(uploaded_image, model)
            max_idx = np.argmax(prediction_scores_temp)
            detected_exercise_temp = CLASS_NAMES[max_idx]
            confidence_val_temp = prediction_scores_temp[max_idx] * 100
            run_prediction = True

        elif uploaded_video is not None:
            st.video(uploaded_video)
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            
            cap = cv2.VideoCapture(tfile.name)
            frame_placeholder = st.empty()
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: 
                    break
                    
                frame_idx += 1
                if frame_idx % 5 == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    
                    prediction_scores_temp = preprocess_and_predict(pil_img, model)
                    max_idx = np.argmax(prediction_scores_temp)
                    detected_exercise_temp = CLASS_NAMES[max_idx]
                    confidence_val_temp = prediction_scores_temp[max_idx] * 100
                    run_prediction = True
                    
                    st.session_state.detected_exercise = detected_exercise_temp
                    st.session_state.confidence_value = confidence_val_temp
                    st.session_state.prediction_scores = prediction_scores_temp
                    st.session_state.frames_analyzed += 1
                    st.session_state.detections_today += 1
                    st.session_state.last_detected_time = time.strftime("%H:%M:%S")
                
                if st.session_state.detected_exercise:
                    cv2.rectangle(frame, (10, 10), (340, 75), (58, 63, 44), -1) 
                    cv2.rectangle(frame, (10, 10), (340, 75), (181, 255, 0), 1) 
                    cv2.putText(frame, f"{st.session_state.detected_exercise.upper()}", (20, 38), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"CONFIDENCE: {st.session_state.confidence_value:.1f}%", (20, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (181, 255, 0), 1, cv2.LINE_AA)
                                
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            cap.release()

        elif use_live_webcam:
            cap = cv2.VideoCapture(0)
            frame_placeholder = st.empty()
            stop_btn = st.button("🔴 STOP STREAM")
            
            frame_idx = 0
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret: 
                    break
                    
                frame_idx += 1
                if frame_idx % 5 == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    
                    prediction_scores_temp = preprocess_and_predict(pil_img, model)
                    max_idx = np.argmax(prediction_scores_temp)
                    detected_exercise_temp = CLASS_NAMES[max_idx]
                    confidence_val_temp = prediction_scores_temp[max_idx] * 100
                    run_prediction = True
                    
                    st.session_state.detected_exercise = detected_exercise_temp
                    st.session_state.confidence_value = confidence_val_temp
                    st.session_state.prediction_scores = prediction_scores_temp
                    st.session_state.frames_analyzed += 1
                    st.session_state.detections_today += 1
                    st.session_state.last_detected_time = time.strftime("%H:%M:%S")
                
                if st.session_state.detected_exercise:
                    cv2.rectangle(frame, (10, 10), (340, 75), (58, 63, 44), -1) 
                    cv2.rectangle(frame, (10, 10), (340, 75), (181, 255, 0), 1) 
                    cv2.putText(frame, f"LIVE: {st.session_state.detected_exercise.upper()}", (20, 38), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"CONFIDENCE: {st.session_state.confidence_value:.1f}%", (20, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (181, 255, 0), 1, cv2.LINE_AA)
                                
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            cap.release()

        # Update Session States on execution
        if run_prediction and (uploaded_image is not None):
            st.session_state.detected_exercise = detected_exercise_temp
            st.session_state.confidence_value = confidence_val_temp
            st.session_state.prediction_scores = prediction_scores_temp
            st.session_state.frames_analyzed += 1
            st.session_state.detections_today += 1
            st.session_state.last_detected_time = time.strftime("%H:%M:%S")

        # Session Stats Row (2x2 metric cards)
        session_duration_sec = int(time.time() - st.session_state.start_time)
        if session_duration_sec < 60:
            duration_str = f"{session_duration_sec}s"
        else:
            duration_str = f"{session_duration_sec // 60}m {session_duration_sec % 60}s"

        st.markdown(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px; margin-bottom: 15px;">
            <div style="background-color: #2C2F33; border: 1px solid #4A4F55; padding: 12px; border-radius: 10px; text-align: center;">
                <span style="font-size: 10px; color: #A0A4A8; text-transform: uppercase; font-weight: 600;">Frames Checked</span>
                <div style="font-size: 1.25rem; font-weight: bold; color: #B5FF00; margin-top: 4px;">{st.session_state.frames_analyzed}</div>
            </div>
            <div style="background-color: #2C2F33; border: 1px solid #4A4F55; padding: 12px; border-radius: 10px; text-align: center;">
                <span style="font-size: 10px; color: #A0A4A8; text-transform: uppercase; font-weight: 600;">Detections Today</span>
                <div style="font-size: 1.25rem; font-weight: bold; color: #B5FF00; margin-top: 4px;">{st.session_state.detections_today}</div>
            </div>
            <div style="background-color: #2C2F33; border: 1px solid #4A4F55; padding: 12px; border-radius: 10px; text-align: center;">
                <span style="font-size: 10px; color: #A0A4A8; text-transform: uppercase; font-weight: 600;">Active Time</span>
                <div style="font-size: 1.25rem; font-weight: bold; color: #B5FF00; margin-top: 4px;">{duration_str}</div>
            </div>
            <div style="background-color: #2C2F33; border: 1px solid #4A4F55; padding: 12px; border-radius: 10px; text-align: center;">
                <span style="font-size: 10px; color: #A0A4A8; text-transform: uppercase; font-weight: 600;">Last Active</span>
                <div style="font-size: 1.25rem; font-weight: bold; color: #B5FF00; margin-top: 4px;">{st.session_state.last_detected_time}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------
    # MIDDLE COLUMN: RESULT CARD
    # ------------------------------------------
    with col2:
        st.markdown("<h3 style='color: #E6E6E6 !important; margin-bottom: 15px;'>🎯 CLASSIFIER RESULT</h3>", unsafe_allow_html=True)
        
        # Render gauge
        gauge_html = draw_svg_gauge(st.session_state.confidence_value)
        st.markdown(gauge_html, unsafe_allow_html=True)
        
        # Prediction Section
        st.markdown("<hr style='border:1px solid #4A4F55; margin: 15px 0;'>", unsafe_allow_html=True)
        
        if st.session_state.detected_exercise:
            icon = CLASS_ICONS.get(st.session_state.detected_exercise, "🏃‍♂️")
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <span style="font-size: 11px; color:#A0A4A8; text-transform: uppercase; font-weight:600; letter-spacing: 0.5px;">Detected Movement</span>
                <div style="font-size: 2.2rem; font-weight:700; color:#E6E6E6; margin-top: 4px;">
                    {icon} {st.session_state.detected_exercise}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sorted Probabilities List
            st.markdown("<span style='font-size: 12px; color:#A0A4A8; text-transform: uppercase; font-weight:600; letter-spacing: 0.5px;'>Probability Spectrum</span>", unsafe_allow_html=True)
            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
            
            sorted_indices = np.argsort(st.session_state.prediction_scores)[::-1]
            for idx in sorted_indices:
                c_name = CLASS_NAMES[idx]
                c_conf = st.session_state.prediction_scores[idx] * 100
                c_icon = CLASS_ICONS.get(c_name, "🏃")
                
                st.markdown(f"""
                <div style="margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; font-size: 13.5px; margin-bottom: 4px;">
                        <span style="color: #E6E6E6;">{c_icon} {c_name}</span>
                        <span style="color: #B5FF00; font-weight: bold;">{c_conf:.1f}%</span>
                    </div>
                    <div style="background-color: #2C2F33; height: 8px; border-radius: 4px; overflow: hidden; border: 1px solid #4A4F55;">
                        <div style="background-color: #B5FF00; width: {c_conf}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Status Line
            st.markdown("<hr style='border:1px solid #4A4F55; margin: 15px 0;'>", unsafe_allow_html=True)
            if st.session_state.confidence_value > 80:
                st.markdown("<p style='font-size: 13px; color:#B5FF00; margin: 0; font-weight: 600; text-align: center;'>Model is performing great! ✅</p>", unsafe_allow_html=True)
            elif st.session_state.confidence_value >= 50:
                st.markdown("<p style='font-size: 13px; color:#A0A4A8; margin: 0; font-weight: 600; text-align: center;'>Telemetry feed quality is stable. ⚡</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='font-size: 13px; color:#FF4D4D; margin: 0; font-weight: 600; text-align: center;'>Low confidence level. Re-align posture. ⚠️</p>", unsafe_allow_html=True)
                
        else:
            st.markdown("""
            <div style="text-align: center; padding: 45px 15px; color: #A0A4A8;">
                <div style="font-size: 32px; margin-bottom: 12px;">📊</div>
                <h5 style="color: #E6E6E6 !important;">Telemetry Stream Idle</h5>
                <p style="font-size: 12.5px; margin: 5px 0 0 0;">Inference telemetry results will list here automatically when feed is active.</p>
            </div>
            """, unsafe_allow_html=True)

    # ------------------------------------------
    # RIGHT COLUMN: AI COACH CARD
    # ------------------------------------------
    with col3:
        st.markdown("<h3 style='color: #E6E6E6 !important; margin-bottom: 15px;'>🧠 AI COACH GUIDE</h3>", unsafe_allow_html=True)
        
        if st.session_state.detected_exercise:
            # Check API Key configuration
            if not groq_api_key:
                st.error("🔑 **Groq API Key Missing!** Please configure `GROQ_API_KEY` inside `.streamlit/secrets.toml` to access the Coach's guide.")
            else:
                if st.session_state.last_coach_exercise != st.session_state.detected_exercise:
                    exercise_info = get_exercise_details(st.session_state.detected_exercise, groq_api_key)
                    st.session_state.coach_guidance = exercise_info
                    st.session_state.last_coach_exercise = st.session_state.detected_exercise
                
                # Parse response
                desc, benefits, tips = parse_coach_response(st.session_state.coach_guidance)
                
                # Render Clean Headers & Styled HTML
                st.markdown(f"<h4 style='color: #C6FF4D !important; margin-bottom: 12px; font-size: 18px;'>Exercise Guide: {st.session_state.detected_exercise.upper()}</h4>", unsafe_allow_html=True)
                
                # Check if we parsed successfully
                if desc or benefits or tips:
                    # Section: Description
                    display_desc = desc if desc else "Refer to raw guidance below."
                    st.markdown(f'<div style="margin-bottom: 15px;"><span style="font-size: 10.5px; color: #A0A4A8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">What is this exercise?</span><p style="font-size: 14px; margin-top: 4px; color: #E6E6E6; line-height: 1.4;">{display_desc}</p></div>', unsafe_allow_html=True)
                    
                    # Section: Benefits
                    if benefits:
                        benefits_html = "".join([f"<li style='margin-bottom: 4px; font-size: 13.5px; color: #E6E6E6;'>{b}</li>" for b in benefits])
                        st.markdown(f'<div style="margin-bottom: 15px;"><span style="font-size: 10.5px; color: #A0A4A8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Benefits</span><ul style="margin-top: 4px; padding-left: 20px; color: #B5FF00;">{benefits_html}</ul></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="margin-bottom: 15px;"><span style="font-size: 10.5px; color: #A0A4A8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Benefits</span><p style="font-size: 13.5px; color: #A0A4A8; font-style: italic; margin-top: 4px;">Refer to raw guidance below.</p></div>', unsafe_allow_html=True)
                    
                    # Section: Key Tips
                    if tips:
                        tips_html = "".join([
                            f'<div style="display: flex; align-items: flex-start; margin-bottom: 10px;">'
                            f'<div style="background-color: #B5FF00; color: #2C2F33; font-weight: bold; border-radius: 50%; width: 18px; height: 18px; display: flex; align-items: center; justify-content: center; margin-right: 10px; flex-shrink: 0; font-size: 10px;">{i+1}</div>'
                            f'<div style="font-size: 13.5px; color: #E6E6E6; line-height: 1.35;">{t}</div>'
                            f'</div>'
                            for i, t in enumerate(tips)
                        ])
                        st.markdown(f'<div style="margin-bottom: 15px;"><span style="font-size: 10.5px; color: #A0A4A8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; display: block; margin-bottom: 6px;">Key Coaching Tips</span>{tips_html}</div>', unsafe_allow_html=True)
                
                # Fallback
                if not desc or not benefits or not tips:
                    st.markdown(f'<div style="margin-top: 15px; margin-bottom: 15px; padding: 12px; background-color: #2C2F33; border: 1px solid #4A4F55; border-radius: 8px;"><span style="font-size: 10.5px; color: #A0A4A8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; display: block; margin-bottom: 6px;">📋 Raw Coach Guidance</span><div style="font-size: 13px; color: #E6E6E6; line-height: 1.4; white-space: pre-wrap;">{st.session_state.coach_guidance}</div></div>', unsafe_allow_html=True)
                
                # Encouragement Message Box
                motivational_msg = get_encouragement_message(st.session_state.detected_exercise)
                st.markdown(f'<div style="background-color: #2C2F33; border: 1px solid #4A4F55; border-radius: 8px; padding: 12px; margin-top: 15px; margin-bottom: 15px;"><span style="font-weight: 700; color: #B5FF00; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">💡 Motivation Corner</span><p style="margin: 4px 0 0 0; font-size: 13px; color: #E6E6E6; font-style: italic; line-height: 1.3;">"{motivational_msg}"</p></div>', unsafe_allow_html=True)
                
                # Download Button
                download_content = f"""AI COACH FITNESS GUIDANCE // FITSENSE AI
==========================================
Detected Exercise: {st.session_state.detected_exercise}

WHAT IS THIS EXERCISE?
{desc}

BENEFITS:
{chr(10).join(['- ' + b for b in benefits])}

KEY TIPS FOR FORM & SAFETY:
{chr(10).join([f'{idx+1}. ' + tip for idx, tip in enumerate(tips)])}
"""
                st.download_button(
                    label="📥 Download Coach's Response",
                    data=download_content,
                    file_name=f"{st.session_state.detected_exercise.lower().replace('-','_')}_guide.txt",
                    mime="text/plain",
                    key="download_guide_btn"
                )
                
        else:
            st.markdown("""
            <div style="text-align: center; padding: 60px 15px; border: 2px dashed #4A4F55; border-radius: 12px; color: #A0A4A8;">
                <div style="font-size: 32px; margin-bottom: 12px;">🤸‍♂️</div>
                <h5 style="color: #E6E6E6 !important;">Coach Mode Offline</h5>
                <p style="font-size: 12.5px; margin: 5px 0 0 0;">Feed video/image frames in the <b>Input Portal</b> (left pane) to initiate the coaching guideline system.</p>
            </div>
            """, unsafe_allow_html=True)

with finder_tab:
    # ------------------------------------------
    # WORKOUT FINDER TAB
    # ------------------------------------------
    st.markdown("<h2 style='color: #E6E6E6 !important; margin-bottom: 20px;'>🎯 Find Your Workout</h2>", unsafe_allow_html=True)
    
    workouts_dict = load_muscle_workouts()
    
    muscle_map = {
        "Chest": "chest",
        "Back": "back",
        "Legs": "legs",
        "Core": "core",
        "Shoulders": "shoulders",
        "Full Body": "full_body"
    }
    
    selected_label = st.selectbox(
        "Which muscle group do you want to work on?",
        list(muscle_map.keys())
    )
    selected_key = muscle_map[selected_label]
    exercise_list = workouts_dict.get(selected_key, [])
    
    if exercise_list:
        if not groq_api_key:
            st.error("🔑 **Groq API Key Missing!** Please configure `GROQ_API_KEY` inside `.streamlit/secrets.toml` to access the Workout Finder.")
        else:
            # Request routine from LLM (cache by selected muscle group key to avoid re-calls)
            routine_cache_key = f"routine_{selected_key}"
            if routine_cache_key not in st.session_state:
                with st.spinner("AI Coach building custom routine..."):
                    routine_raw = generate_routine(selected_label, exercise_list, groq_api_key)
                    st.session_state[routine_cache_key] = routine_raw
                    
            routine_data = st.session_state[routine_cache_key]
            
            # Temporary debug print to output raw LLM response for Shoulders routine
            if selected_key == "shoulders":
                print(f"[DEBUG] Raw Shoulders Routine LLM Response:\n{routine_data}")
                
            warmup, routine_items, cooldown = parse_routine_response(routine_data)
            
            # Styled routine container
            st.markdown(f'<div style="background-color: #3A3F44; border: 1px solid #4A4F55; border-radius: 14px; padding: 24px; box-shadow: 0px 8px 30px rgba(0, 0, 0, 0.2); margin-bottom: 20px;"><h3 style="color: #B5FF00 !important; margin-bottom: 15px; font-size: 20px;">📋 Custom {selected_label} Workout</h3><div style="margin-bottom: 20px;"><span style="font-size: 11px; color: #A0A4A8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">🔥 Warm-up Routine</span><p style="font-size: 14.5px; margin-top: 4px; color: #E6E6E6; line-height: 1.45;">{warmup}</p></div><div style="margin-bottom: 20px;"><span style="font-size: 11px; color: #A0A4A8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">❄️ Recovery Cool-Down</span><p style="font-size: 14.5px; margin-top: 4px; color: #E6E6E6; line-height: 1.45;">{cooldown}</p></div></div>', unsafe_allow_html=True)
            
            # Exercises
            st.markdown("<h4 style='color: #E6E6E6 !important; margin-top: 20px; font-size: 16px;'>🏃 Exercises:</h4>", unsafe_allow_html=True)
            guide_dict = load_exercise_guide()
            
            # Helper function for robust exercise-to-item matching
            def is_exercise_in_item(ex_name, item_text):
                # 1. Direct case-insensitive containment
                if ex_name.lower() in item_text.lower():
                    return True
                # 2. Alphanumeric normalized comparison (removes hyphens, spaces, symbols)
                ex_norm = re.sub(r'[^a-z0-9]', '', ex_name.lower())
                item_norm = re.sub(r'[^a-z0-9]', '', item_text.lower())
                if ex_norm in item_norm:
                    return True
                # 3. Word-level containment (checks if all words in exercise name are present in the item words)
                ex_words = [w for w in re.split(r'[^a-z0-9]', ex_name.lower()) if w]
                item_words = set(re.split(r'[^a-z0-9]', item_text.lower()))
                if ex_words and all(any(w in iw for iw in item_words) for w in ex_words):
                    return True
                return False
            
            unique_matched_exercises = []
            for item in routine_items:
                for ex_key in guide_dict.keys():
                    if is_exercise_in_item(ex_key, item):
                        if ex_key not in unique_matched_exercises:
                            unique_matched_exercises.append(ex_key)
            
            exercise_to_items = {ex: [] for ex in unique_matched_exercises}
            
            # Fallback checking: Ensure all exercises in the current muscle group's exercise_list are represented
            for exercise_name in exercise_list:
                # Find matching key in guide_dict (case-insensitively)
                matched_key = None
                for ex_key in guide_dict.keys():
                    if ex_key.lower() == exercise_name.lower():
                        matched_key = ex_key
                        break
                if not matched_key:
                    matched_key = exercise_name
                
                # Check if this exercise key is in unique_matched_exercises
                is_matched = False
                for matched_ex in unique_matched_exercises:
                    if matched_ex.lower() == matched_key.lower():
                        is_matched = True
                        break
                
                if not is_matched:
                    unique_matched_exercises.append(matched_key)
                    if matched_key not in exercise_to_items:
                        exercise_to_items[matched_key] = []
                    # Add generic sets/reps
                    exercise_to_items[matched_key].append(f"{matched_key} - 3 sets of 10-12 reps")
            
            unmatched_items = []
            for item in routine_items:
                # Filter out the default/vague filler sentence
                if "Perform core movements cleanly." in item:
                    continue
                    
                matched_any = False
                for ex in unique_matched_exercises:
                    if is_exercise_in_item(ex, item):
                        if ex not in exercise_to_items:
                            exercise_to_items[ex] = []
                        exercise_to_items[ex].append(item)
                        matched_any = True
                if not matched_any:
                    unmatched_items.append(item)
            
            for ex in unique_matched_exercises:
                items = exercise_to_items[ex]
                if not items:
                    continue
                
                header_item = items[0]
                tips_items = items[1:]
                
                # 1. EXERCISE HEADER CARD
                st.markdown(f'<div style="background-color: #3A3F44; border: 1px solid #4A4F55; border-radius: 8px; padding: 14px; margin-top: 15px; margin-bottom: 5px; box-shadow: 0 4px 6px rgba(181, 255, 0, 0.05);"><span style="font-size: 15px; color: #B5FF00; font-weight: 700;">{header_item}</span></div>', unsafe_allow_html=True)
                
                # 2. TIPS/NOTES CARD
                if tips_items:
                    tips_html = "".join([f"<p style='margin: 0 0 6px 0; font-size: 13px; color: #A0A4A8; line-height: 1.4;'>💡 {tip}</p>" for tip in tips_items])
                    st.markdown(f'<div style="background-color: #2C2F33; border: 1px solid #4A4F55; border-radius: 8px; padding: 12px; margin-bottom: 8px;">{tips_html}</div>', unsafe_allow_html=True)
                
                # 3. "HOW TO PERFORM" EXPANDER
                with st.expander(f"▶ How to perform: {ex}"):
                    ex_info = guide_dict[ex]
                    render_exercise_card(ex, ex_info)
                
                st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)
            
            for item in unmatched_items:
                st.markdown(f'<div style="background-color: #3A3F44; border: 1px solid #4A4F55; border-radius: 8px; padding: 12px; margin-bottom: 5px;"><span style="font-size: 14px; color: #E6E6E6; font-weight: 500;">{item}</span></div>', unsafe_allow_html=True)

with rehab_tab:
    # ------------------------------------------
    # REHAB GUIDE TAB
    # ------------------------------------------
    st.markdown("<h2 style='color: #E6E6E6 !important; margin-bottom: 20px;'>🩹 Rehab Guide</h2>", unsafe_allow_html=True)
    
    # 1. Disclaimer Banner
    st.markdown("""
    <div style="background-color: #3A3F44; border-left: 5px solid #FF4D4D; border-top: 1px solid #4A4F55; border-right: 1px solid #4A4F55; border-bottom: 1px solid #4A4F55; padding: 15px; border-radius: 8px; margin-bottom: 25px;">
        <span style="font-weight: 700; color: #FF4D4D; font-size: 13px; display: block; margin-bottom: 4px;">⚠️ MEDICAL DISCLAIMER</span>
        <p style="margin: 0; font-size: 13.5px; color: #E6E6E6; line-height: 1.4;">
            This provides general guidance only and does not replace professional medical advice. If pain is severe, sudden, or worsening, consult a doctor or physiotherapist.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Triage Question
    triage_choice = st.radio(
        "How would you describe this?",
        [
            "Mild stiffness or soreness",
            "Sharp pain, swelling, numbness, or a diagnosed injury"
        ]
    )
    
    # 3. Severe Triage Path
    if triage_choice == "Sharp pain, swelling, numbness, or a diagnosed injury":
        st.markdown("""
        <div style="background-color: #3A3F44; border: 1px solid #FF4D4D; border-radius: 12px; padding: 25px; text-align: center; margin-top: 20px;">
            <div style="font-size: 48px; margin-bottom: 15px;">🚨</div>
            <h4 style="color: #FF4D4D !important; font-weight: 700; margin-bottom: 10px;">Immediate Care Required</h4>
            <p style="color: #E6E6E6; font-size: 14.5px; margin: 0 auto; max-width: 480px; line-height: 1.5;">
                We strongly advise that you <b>consult a doctor, physiotherapist, or medical professional immediately</b>. 
                Do not attempt self-guided exercises or training movements as this can aggravate a potentially serious injury.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # 4. Mild Triage Path
        selected_area_label = st.selectbox(
            "Which area?",
            ["Lower Back", "Knee", "Shoulder", "Neck", "Ankle"]
        )
        
        rehab_map = {
            "Lower Back": "lower_back",
            "Knee": "knee",
            "Shoulder": "shoulder",
            "Neck": "neck",
            "Ankle": "ankle"
        }
        
        rehab_data = load_rehab_exercises()
        mapped_key = rehab_map[selected_area_label]
        
        area_details = rehab_data.get(mapped_key, {})
        exercise_list = area_details.get("exercises", [])
        avoid_list = area_details.get("avoid", [])
        severity_note = area_details.get("severity_note", "")
        
        if exercise_list:
            if severity_note:
                st.info(f"💡 **Note**: {severity_note}")
                
            # Recommended exercises expanders
            st.markdown("<h4 style='color: #E6E6E6 !important; margin-top: 15px; font-size: 16px;'>🩹 Recommended Rehab Movements:</h4>", unsafe_allow_html=True)
            guide_dict = load_exercise_guide()
            
            unique_rehab_exercises = []
            for ex_name in exercise_list:
                for ex_key in guide_dict.keys():
                    if ex_key.lower() == ex_name.lower():
                        if ex_key not in unique_rehab_exercises:
                            unique_rehab_exercises.append(ex_key)
            
            for ex in unique_rehab_exercises:
                # 1. EXERCISE HEADER CARD
                st.markdown(f'<div style="background-color: #3A3F44; border: 1px solid #B5FF00; border-radius: 8px; padding: 14px; margin-top: 15px; margin-bottom: 5px; box-shadow: 0 4px 6px rgba(181, 255, 0, 0.05);"><span style="font-size: 15px; color: #B5FF00; font-weight: 700;">{ex}</span></div>', unsafe_allow_html=True)
                
                # 3. "HOW TO PERFORM" EXPANDER
                with st.expander(f"▶ How to perform: {ex}"):
                    ex_info = guide_dict[ex]
                    render_exercise_card(ex, ex_info)
                        
                st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)
                            
            # 5. Pass to Groq/LangChain
            if not groq_api_key:
                st.error("🔑 **Groq API Key Missing!** Please configure `GROQ_API_KEY` inside `.streamlit/secrets.toml` to access the Rehab Guide.")
            else:
                rehab_cache_key = f"rehab_{mapped_key}"
                if rehab_cache_key not in st.session_state:
                    with st.spinner("AI Coach analyzing rehab path..."):
                        rehab_raw = generate_rehab_guidance(selected_area_label, exercise_list, groq_api_key)
                        st.session_state[rehab_cache_key] = rehab_raw
                        
                guidance_text = st.session_state[rehab_cache_key]
                
                # 6. Display response in a card
                st.markdown(f'<div style="background-color: #3A3F44; border: 1px solid #4A4F55; border-radius: 14px; padding: 24px; box-shadow: 0px 8px 30px rgba(0, 0, 0, 0.2); margin-top: 20px; margin-bottom: 20px;"><h3 style="color: #B5FF00 !important; margin-bottom: 15px; font-size: 20px;">🩹 Gentle Rehab Path: {selected_area_label}</h3><div style="color: #E6E6E6; font-size: 14.5px; line-height: 1.55;">{guidance_text}</div></div>', unsafe_allow_html=True)
                
                # Avoid list
                avoid_items_html = "".join([f"<li style='margin-bottom: 4px; font-size: 13.5px; color: #E6E6E6;'>{item}</li>" for item in avoid_list])
                st.markdown(f'<div style="background-color: #3A3F44; border-left: 5px solid #FF4D4D; border-top: 1px solid #4A4F55; border-right: 1px solid #4A4F55; border-bottom: 1px solid #4A4F55; padding: 18px; border-radius: 8px;"><span style="font-weight: 700; color: #FF4D4D; font-size: 13px; display: block; margin-bottom: 8px;">⚠️ Avoid These While Recovering</span><ul style="margin: 0; padding-left: 20px; color: #FF4D4D;">{avoid_items_html}</ul></div>', unsafe_allow_html=True)

# ==========================================
# 11. CALORIE & PROTEIN CALCULATOR TAB
# ==========================================
with calc_tab:
    st.markdown("<h2 style='color: #E6E6E6 !important; margin-bottom: 5px;'>🔥 CALORIE & PROTEIN CALCULATOR</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #A0A4A8; font-size: 15px; margin-bottom: 20px;'>Calculate your BMR, TDEE, goal daily calories, and protein target using the Mifflin-St Jeor equation.</p>", unsafe_allow_html=True)

    # Disclaimer Banner
    st.markdown("""
    <div style="background-color: #3A3F44; border-left: 5px solid #FF4D4D; border-top: 1px solid #4A4F55; border-right: 1px solid #4A4F55; border-bottom: 1px solid #4A4F55; padding: 16px; border-radius: 8px; margin-bottom: 25px;">
        <span style="font-weight: 700; color: #FF4D4D; font-size: 14px; display: block; margin-bottom: 4px;">⚠️ MEDICAL & NUTRITION DISCLAIMER</span>
        <p style="margin: 0; font-size: 13.5px; color: #E6E6E6;">
            These calculations and meal plan recommendations are estimates based on standard formulas (Mifflin-St Jeor) and general guidelines. They are not a substitute for professional medical or nutrition advice. Consult a doctor or registered dietitian for specific dietary requirements or health conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 1. User Input Form
    st.markdown("<h3 style='color: #E6E6E6 !important; margin-bottom: 15px;'>📋 BIOMETRICS & GOALS</h3>", unsafe_allow_html=True)
    
    col_in1, col_in2 = st.columns(2, gap="large")

    with col_in1:
        st.markdown("<h4 style='color: #B5FF00 !important; font-size: 16px; margin-bottom: 10px;'>1. Personal Details & Units</h4>", unsafe_allow_html=True)
        age = st.number_input("Age (years)", min_value=15, max_value=80, value=25, step=1, key="calc_age")
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True, key="calc_gender")
        unit_toggle = st.radio("Unit System", ["Metric (cm / kg)", "US (ft+in / lbs)"], horizontal=True, key="calc_unit_toggle")

        if unit_toggle == "Metric (cm / kg)":
            height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=175.0, step=0.5, key="calc_h_cm")
            weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.5, key="calc_w_kg")
        else:
            col_ft, col_in = st.columns(2)
            height_ft = col_ft.number_input("Height (feet)", min_value=1, max_value=8, value=5, step=1, key="calc_h_ft")
            height_in = col_in.number_input("Height (inches)", min_value=0.0, max_value=11.9, value=9.0, step=0.5, key="calc_h_in")
            height_cm = (height_ft * 12 + height_in) * 2.54

            weight_lbs = st.number_input("Weight (lbs)", min_value=44.0, max_value=660.0, value=154.0, step=1.0, key="calc_w_lbs")
            weight_kg = weight_lbs / 2.20462

    with col_in2:
        st.markdown("<h4 style='color: #B5FF00 !important; font-size: 16px; margin-bottom: 10px;'>2. Lifestyle & Objectives</h4>", unsafe_allow_html=True)
        
        activity_options = {
            "Sedentary (little or no exercise)": 1.2,
            "Light (exercise 1–3 times/week)": 1.375,
            "Moderate (exercise 4–5 times/week)": 1.55,
            "Active (daily exercise or intense exercise 3–4 times/week)": 1.725,
            "Very Active (intense exercise 6–7 times/week)": 1.9,
            "Extra Active (very intense exercise daily or physical job)": 1.95
        }
        
        selected_activity_label = st.selectbox("Activity Level", list(activity_options.keys()), index=2, key="calc_activity")
        activity_multiplier = activity_options[selected_activity_label]

        goal = st.selectbox("Primary Goal", ["Lose Weight", "Maintain Weight", "Gain Muscle"], key="calc_goal")
        fitness_goal_protein = st.radio("Fitness Goal for Protein", ["General Health", "Muscle Building"], horizontal=True, key="calc_protein_goal")

    # 2. Calorie Calculation Logic (Mifflin-St Jeor)
    if gender == "Male":
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

    tdee = bmr * activity_multiplier

    caution_note = None
    if goal == "Lose Weight":
        goal_calories = tdee - 500
        explanation_line = "Mild deficit (~500 kcal/day) targeted to safely lose approx 1 lb (0.45 kg) per week."
        if gender == "Female" and goal_calories < 1200:
            caution_note = "⚠️ Note: Resulting intake is below 1,200 kcal/day (recommended minimum for women per Harvard Health). Consider a smaller deficit."
        elif gender == "Male" and goal_calories < 1500:
            caution_note = "⚠️ Note: Resulting intake is below 1,500 kcal/day (recommended minimum for men per Harvard Health). Consider a smaller deficit."
    elif goal == "Maintain Weight":
        goal_calories = tdee
        explanation_line = "Maintenance intake equals your Total Daily Energy Expenditure (TDEE) to sustain current weight."
    else:  # Gain Muscle
        goal_calories = tdee + 400
        explanation_line = "Moderate surplus (+400 kcal/day) to support muscle building while minimizing excess fat storage."

    # 3. Protein Calculation Logic
    if fitness_goal_protein == "General Health":
        protein_min = 0.8 * weight_kg
        protein_max = 1.0 * weight_kg
        protein_target = 0.9 * weight_kg
    else:  # Muscle Building
        protein_min = 1.6 * weight_kg
        protein_max = 2.2 * weight_kg
        protein_target = 1.8 * weight_kg

    # Render Calorie & Protein Results Cards
    st.markdown("<h3 style='color: #E6E6E6 !important; margin-top: 25px; margin-bottom: 15px;'>🎯 YOUR CALORIE & PROTEIN TARGETS</h3>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 15px;">
        <div style="background-color: #2C2F33; border: 1px solid #4A4F55; padding: 18px; border-radius: 12px; text-align: center;">
            <span style="font-size: 11px; color: #A0A4A8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">BMR (Basal Rate)</span>
            <div style="font-size: 1.6rem; font-weight: bold; color: #E6E6E6; margin-top: 6px;">{int(round(bmr))} <span style="font-size: 1rem; color: #A0A4A8;">kcal</span></div>
        </div>
        <div style="background-color: #2C2F33; border: 1px solid #4A4F55; padding: 18px; border-radius: 12px; text-align: center;">
            <span style="font-size: 11px; color: #A0A4A8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Maintenance (TDEE)</span>
            <div style="font-size: 1.6rem; font-weight: bold; color: #E6E6E6; margin-top: 6px;">{int(round(tdee))} <span style="font-size: 1rem; color: #A0A4A8;">kcal</span></div>
        </div>
        <div style="background-color: #2C2F33; border: 1px solid #B5FF00; padding: 18px; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(181, 255, 0, 0.1);">
            <span style="font-size: 11px; color: #B5FF00; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Goal Calorie Target</span>
            <div style="font-size: 1.6rem; font-weight: bold; color: #B5FF00; margin-top: 6px;">{int(round(goal_calories))} <span style="font-size: 1rem; color: #E6E6E6;">kcal/day</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background-color: #2C2F33; border: 1px solid #4A4F55; padding: 14px 18px; border-radius: 10px; margin-top: 10px; margin-bottom: 15px;">
        <span style="color: #E6E6E6; font-size: 14px;">💡 <strong>Explanation</strong>: {explanation_line}</span>
    </div>
    """, unsafe_allow_html=True)

    if caution_note:
        st.warning(caution_note)

    # Protein Card Banner
    st.markdown(f"""
    <div style="background-color: #3A3F44; border: 1px solid #4A4F55; padding: 20px; border-radius: 12px; margin-top: 15px; margin-bottom: 25px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="font-size: 12px; color: #A0A4A8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Recommended Daily Protein Range</span>
                <h3 style="color: #E6E6E6 !important; margin: 4px 0 0 0; font-size: 1.3rem;">
                    You need approximately <span style="color: #A0A4A8; font-weight: 600;">{int(round(protein_min))}g – {int(round(protein_max))}g</span> per day
                </h3>
            </div>
            <div style="background-color: #2C2F33; border: 1px solid #B5FF00; padding: 10px 18px; border-radius: 8px; margin-top: 8px; box-shadow: 0 4px 15px rgba(181, 255, 0, 0.08);">
                <span style="font-size: 11px; color: #B5FF00; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Optimal Target</span>
                <div style="font-size: 1.25rem; font-weight: bold; color: #B5FF00;">~{int(round(protein_target))}g / day</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4. Protein Sources Reference
    st.markdown("<h3 style='color: #E6E6E6 !important; margin-top: 30px; margin-bottom: 15px;'>🥗 PROTEIN SOURCES REFERENCE</h3>", unsafe_allow_html=True)
    
    diet_preference = st.selectbox("Choose your diet type", ["Vegetarian", "Non-Vegetarian"], key="calc_diet_pref")

    protein_data = load_protein_sources()
    veg_sources = protein_data.get("vegetarian", [])
    nonveg_sources = protein_data.get("non_vegetarian", [])

    if diet_preference == "Vegetarian":
        selected_sources = veg_sources
        header_title = "🥦 Vegetarian Sources"
    else:
        selected_sources = nonveg_sources
        header_title = "🍗 Non-Vegetarian Sources"

    st.markdown(f"""
    <div style="background-color: #3A3F44; border: 1px solid #B5FF00; border-radius: 12px; padding: 18px; margin-top: 10px; margin-bottom: 15px;">
        <h4 style="color: #E6E6E6 !important; font-size: 16px; margin: 0;">{header_title} <span style='font-size: 11px; background-color: #B5FF00; color: #2C2F33; padding: 2px 8px; border-radius: 10px; font-weight: bold;'>YOUR PREFERENCE</span></h4>
    </div>
    """, unsafe_allow_html=True)

    for item in selected_sources:
        st.markdown(f"""
        <div style="background-color: #2C2F33; border: 1px solid #4A4F55; padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong style="color: #E6E6E6; font-size: 14px;">{item['food']}</strong>
                <div style="color: #A0A4A8; font-size: 12px;">Serving: {item['serving']}</div>
            </div>
            <div style="background-color: #3A3F44; color: #B5FF00; font-weight: bold; padding: 4px 10px; border-radius: 6px; font-size: 13px;">
                {item['protein_g']}g protein
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 5. AI-Generated Meal Plan Section
    st.markdown("<h3 style='color: #E6E6E6 !important; margin-top: 30px; margin-bottom: 15px;'>🤖 AI-GENERATED MEAL PLAN</h3>", unsafe_allow_html=True)
    
    if not groq_api_key:
        st.error("🔑 **Groq API Key Missing!** Please configure `GROQ_API_KEY` inside `.streamlit/secrets.toml` to generate an AI meal plan.")
    else:
        protein_context_str = ", ".join([f"{item['food']} ({item['protein_g']}g protein / {item['serving']})" for item in selected_sources])
        
        meal_plan_cache_key = f"meal_plan_{goal}_{int(round(goal_calories))}_{int(round(protein_target))}_{diet_preference}"
        
        if st.button("✨ Generate My Meal Plan", key="generate_meal_plan_btn"):
            with st.spinner("AI Coach crafting your 1-day sample meal plan..."):
                plan_raw = generate_meal_plan(
                    goal=goal,
                    calorie_target=int(round(goal_calories)),
                    protein_target=int(round(protein_target)),
                    diet_pref=diet_preference,
                    protein_sources_context=protein_context_str,
                    api_key=groq_api_key
                )
                st.session_state[meal_plan_cache_key] = plan_raw

        if meal_plan_cache_key in st.session_state:
            plan_text = st.session_state[meal_plan_cache_key]
            
            breakfast, lunch, snacks, dinner, summary_text, disclaimer_text = parse_meal_plan_response(plan_text)
            
            st.markdown(f"""
            <div style="background-color: #3A3F44; border: 1px solid #4A4F55; border-radius: 14px; padding: 24px; margin-top: 15px; margin-bottom: 20px;">
                <h3 style="color: #B5FF00 !important; margin-bottom: 15px; font-size: 20px;">📋 Sample 1-Day Meal Plan ({diet_preference})</h3>
            """, unsafe_allow_html=True)

            meals_data = [
                ("🌅 Breakfast", breakfast),
                ("☀️ Lunch", lunch),
                ("🍎 Snacks", snacks),
                ("🌙 Dinner", dinner)
            ]

            col_m1, col_m2 = st.columns(2, gap="medium")
            for idx, (meal_title, meal_items) in enumerate(meals_data):
                target_col = col_m1 if idx % 2 == 0 else col_m2
                with target_col:
                    st.markdown(f"""
                    <div style="background-color: #2C2F33; border: 1px solid #4A4F55; padding: 14px; border-radius: 10px; margin-bottom: 15px;">
                        <h4 style="color: #B5FF00 !important; font-size: 15px; margin-bottom: 8px;">{meal_title}</h4>
                    """, unsafe_allow_html=True)
                    if meal_items:
                        for item in meal_items:
                            st.write(f"• {item}")
                    else:
                        st.write("• Balanced serving according to calorie/protein target.")
                    st.markdown("</div>", unsafe_allow_html=True)

            if summary_text:
                st.markdown(f"""
                <div style="background-color: #2C2F33; border: 1px solid #B5FF00; padding: 14px; border-radius: 10px; margin-top: 10px; margin-bottom: 15px;">
                    <span style="color: #B5FF00; font-weight: bold;">📊 Daily Summary:</span>
                    <p style="color: #E6E6E6; margin: 4px 0 0 0; font-size: 14px;">{summary_text}</p>
                </div>
                """, unsafe_allow_html=True)

            if disclaimer_text:
                st.caption(f"⚕️ Note: {disclaimer_text}")

            st.markdown("</div>", unsafe_allow_html=True)

            st.download_button(
                label="📥 Download Meal Plan (.txt)",
                data=plan_text,
                file_name=f"meal_plan_{diet_preference.lower()}_{int(round(goal_calories))}kcal.txt",
                mime="text/plain",
                key="download_meal_plan_btn"
            )

