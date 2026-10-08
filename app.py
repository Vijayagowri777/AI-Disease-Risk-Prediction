import streamlit as st
import pandas as pd
import joblib
import os
import time

from dotenv import load_dotenv
from google import genai


# =========================================================
# GEMINI CONFIGURATION
# =========================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

if api_key:
    client = genai.Client(api_key=api_key)
else:
    client = None


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Health Risk Assessment",
    page_icon="🩺",
    layout="centered"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #F7F9FC;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* ---------- Header ---------- */

.app-header {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 1rem;
}

.app-icon {
    font-size: 2.4rem;
    line-height: 1;
}

.app-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #172033;
    margin: 0;
}


/* ---------- Section headings ---------- */

.section-title {
    font-size: 1.25rem;
    font-weight: 650;
    color: #172033;
    margin-top: 0.8rem;
    margin-bottom: 0.3rem;
}

.section-subtitle {
    font-size: 0.95rem;
    color: #64748B;
    margin-bottom: 1.2rem;
}


/* ---------- Text inputs ---------- */

div[data-testid="stTextInput"] label {
    color: #172033 !important;
    font-weight: 650 !important;
    font-size: 0.95rem !important;
}

div[data-testid="stTextInput"] input {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    color: #172033 !important;
    height: 42px !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 1px #2563EB !important;
}


/* ---------- Select box ---------- */

div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    color: #172033 !important;
    min-height: 42px !important;
}


div[data-baseweb="select"] span {
    color: #172033 !important;
}


/* ---------- Dropdown menu ---------- */

div[data-baseweb="popover"] {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
}

div[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
}

div[data-baseweb="menu"] li {
    background-color: #FFFFFF !important;
    color: #172033 !important;
}

div[data-baseweb="menu"] li:hover {
    background-color: #E2E8F0 !important;
    color: #172033 !important;
}

div[data-baseweb="menu"] li[aria-selected="true"] {
    background-color: #EFF6FF !important;
    color: #172033 !important;
}


/* ---------- Main button ---------- */

div.stButton > button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    height: 46px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: 0.2s ease;
}

div.stButton > button:hover {
    background-color: #1D4ED8 !important;
}


/* ---------- Result card ---------- */

.result-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.8rem 1.5rem;
    margin-top: 2rem;
    margin-bottom: 1.5rem;

    display: flex;
    justify-content: space-between;
    gap: 2rem;
}

.result-column {
    flex: 1;
}

.result-divider {
    border-left: 1px solid #E2E8F0;
    padding-left: 2rem;
}

.result-label {
    color: #64748B;
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
}

.result-value {
    color: #172033;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.1;
}

.risk-category {
    font-size: 1.15rem;
    font-weight: 650;
}

.lower-risk {
    color: #15803D;
}

.higher-risk {
    color: #DC2626;
}


/* ---------- Gemini guidance ---------- */

.guidance-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
    color: #172033;
    line-height: 1.7;
    font-size: 0.95rem;
}


/* ---------- Error message ---------- */

div[data-testid="stAlert"] {
    border-radius: 8px;
}


/* ---------- Hide Streamlit footer ---------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_diabetes_model():
    return joblib.load("models/diabetes_model.pkl")


@st.cache_resource
def load_heart_model():
    return joblib.load("models/heart_model.pkl")


diabetes_model = load_diabetes_model()
heart_model = load_heart_model()


# =========================================================
# GEMINI HEALTH GUIDANCE
# =========================================================

def generate_health_guidance(
    disease,
    risk_percentage,
    category,
    patient_data
):

    if client is None:
        return None

    prompt = f"""
You are a health education assistant inside a health risk assessment application.

A machine learning model has estimated a user's risk for {disease}.

Estimated model risk: {risk_percentage:.0f}%
Risk category: {category}

User-provided information:
{patient_data}

Write a short, clear and helpful explanation for the user.

Follow these rules strictly:

- This is a model-based risk estimate, NOT a medical diagnosis.
- Do not diagnose the user.
- Do not claim that the user has or does not have {disease}.
- Do not prescribe medicines.
- Do not recommend starting, stopping, or changing medication.
- Explain that the estimate is based on the information entered and patterns learned from the training dataset.
- Explain the result in simple, non-technical language.
- Give general health education and reasonable next steps.
- Encourage the user to discuss concerns or abnormal measurements with a qualified healthcare professional.
- Do not create unnecessary fear or anxiety.
- Do not use complicated machine-learning terminology.
- Do not mention that you are Gemini.
- Do not use headings such as "What this means", "What you can do next", or "Gemini explanation".
- Keep the response concise.
"""

    for attempt in range(2):

        try:

            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

            return response.text

        except Exception as e:

            error_message = str(e)

            if "503" in error_message or "UNAVAILABLE" in error_message:

                if attempt == 0:
                    time.sleep(3)
                    continue

            break

    return None


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
<div class="app-header">
    <div class="app-icon">🩺</div>
    <div class="app-title">Health Risk Assessment</div>
</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# HEALTH CONDITION
# =========================================================

st.markdown(
    '<div class="section-title">Health condition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">What would you like to assess?</div>',
    unsafe_allow_html=True
)

disease = st.selectbox(
    "Condition",
    ["Diabetes", "Heart Disease"],
    label_visibility="collapsed"
)


# =========================================================
# DIABETES FORM
# =========================================================

if disease == "Diabetes":

    st.markdown(
        '<div class="section-title">Diabetes</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Enter your health information</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # ROW 1
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        age_input = st.text_input(
            "Age",
            placeholder="Enter your age"
        )

    with col2:

        pregnancies_input = st.text_input(
            "Pregnancies",
            placeholder="Enter number of pregnancies"
        )


    # -----------------------------------------------------
    # ROW 2
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        glucose_input = st.text_input(
            "Blood glucose",
            placeholder="e.g. 140 mg/dL",
            help="Blood glucose measurement from a blood test."
        )

    with col2:

        blood_pressure_input = st.text_input(
            "Blood pressure",
            placeholder="e.g. 80 mmHg",
            help="Enter your blood pressure measurement."
        )


    # -----------------------------------------------------
    # ROW 3
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        bmi_input = st.text_input(
            "BMI",
            placeholder="e.g. 32.0",
            help="Body Mass Index."
        )

    with col2:

        insulin_input = st.text_input(
            "Insulin",
            placeholder="Enter insulin level",
            help="Insulin level from a blood test."
        )


    # -----------------------------------------------------
    # ROW 4
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        skin_thickness_input = st.text_input(
            "Skin thickness",
            placeholder="Enter measurement",
            help="Skinfold thickness measurement, usually obtained using a caliper."
        )

    with col2:

        pedigree_input = st.text_input(
            "Family history",
            placeholder="Enter value",
            help="A numerical measure related to diabetes family history in the dataset."
        )


    # -----------------------------------------------------
    # CHECK RISK
    # -----------------------------------------------------

    st.write("")

    check_risk = st.button(
        "Check My Risk →",
        use_container_width=True
    )


    # -----------------------------------------------------
    # DIABETES PREDICTION
    # -----------------------------------------------------

    if check_risk:

        try:

            age = float(age_input)
            pregnancies = float(pregnancies_input)
            glucose = float(glucose_input)
            blood_pressure = float(blood_pressure_input)
            bmi = float(bmi_input)
            insulin = float(insulin_input)
            skin_thickness = float(skin_thickness_input)
            pedigree = float(pedigree_input)

            if age < 1 or age > 100:
                st.error("Please enter a valid age between 1 and 100.")
                st.stop()

            if pregnancies < 0 or pregnancies > 20:
                st.error("Please enter a valid number of pregnancies between 0 and 20.")
                st.stop()
            
            if glucose < 1 or glucose > 300:
                st.error("Please enter a valid blood glucose value between 1 and 300 mg/dL.")
                st.stop()
                
            if blood_pressure < 1 or blood_pressure > 250:
                st.error("Please enter a valid blood pressure value between 1 and 250 mmHg.")
                st.stop()
            
            if bmi < 1 or bmi > 80:
                st.error("Please enter a valid BMI between 1 and 80.")
                st.stop()
            
            if insulin < 0 or insulin > 900:
                st.error("Please enter a valid insulin value between 0 and 900.")
                st.stop()
                
            if skin_thickness < 0 or skin_thickness > 100:
                st.error("Please enter a valid skin thickness between 0 and 100 mm.")
                st.stop()
            
            if pedigree < 0 or pedigree > 3:
                st.error("Please enter a valid family history value between 0 and 3.")
                st.stop()

            patient = pd.DataFrame([{

                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": pedigree,
                "Age": age

            }])


            prediction = diabetes_model.predict(patient)[0]

            probability = diabetes_model.predict_proba(patient)[0][1]

            risk_percentage = probability * 100


            if prediction == 1:

                category = "Higher Risk"
                category_class = "higher-risk"

            else:

                category = "Lower Risk"
                category_class = "lower-risk"


            result_html = f"""
<div class="result-card">

    <div class="result-column">

        <div class="result-label">
            Estimated Risk
        </div>

        <div class="result-value">
            {risk_percentage:.0f}%
        </div>

    </div>

    <div class="result-column result-divider">

        <div class="result-label">
            Category
        </div>

        <div class="risk-category {category_class}">
            ● {category}
        </div>

    </div>

</div>
"""

            st.html(result_html)


            try:

                guidance = generate_health_guidance(
                    disease,
                    risk_percentage,
                    category,
                    patient.to_dict(orient="records")[0]
                )

                if guidance:

                    guidance_html = f"""
<div class="guidance-card">
    {guidance.replace(chr(10), "<br>")}
</div>
"""

                    st.markdown(
                        guidance_html,
                        unsafe_allow_html=True
                    )

                else:

                    st.info(
                        "Additional health guidance is temporarily unavailable."
                    )

            except Exception:

                st.info(
                    "The risk assessment was completed, "
                    "but additional health guidance is temporarily unavailable."
                )


        except ValueError:

            st.error(
                "Please enter valid numeric values in all fields."
            )

        except Exception as e:

            st.error(
                f"Something went wrong while making the assessment: {e}"
            )


# =========================================================
# HEART DISEASE FORM
# =========================================================

else:

    st.markdown(
        '<div class="section-title">Heart Disease</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Enter your health information</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # ROW 1
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        sex_input = st.selectbox(
            "Sex",
            ["Female", "Male"]
        )

    with col2:

        chest_pain_input = st.selectbox(
            "Chest pain type",
            [
                "Typical angina",
                "Atypical angina",
                "Non-anginal pain",
                "Asymptomatic"
            ],
            help="Select the type of chest pain or chest discomfort reported."
        )


    # -----------------------------------------------------
    # ROW 2
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        heart_bp_input = st.text_input(
            "Resting blood pressure",
            placeholder="e.g. 130 mmHg",
            help="Your blood pressure measurement while resting."
        )

    with col2:

        cholesterol_input = st.text_input(
            "Cholesterol",
            placeholder="e.g. 220 mg/dL",
            help="Serum cholesterol measurement from a blood test."
        )


    # -----------------------------------------------------
    # ROW 3
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        max_heart_rate_input = st.text_input(
            "Maximum heart rate",
            placeholder="e.g. 150 bpm",
            help="Maximum heart rate achieved during an exercise test."
        )

    with col2:

        exercise_angina_input = st.selectbox(
            "Exercise-induced angina",
            ["No", "Yes"],
            help="Whether chest discomfort/angina occurred during exercise."
        )


    # -----------------------------------------------------
    # CHECK RISK
    # -----------------------------------------------------

    st.write("")

    check_heart_risk = st.button(
        "Check My Risk →",
        use_container_width=True
    )


    # -----------------------------------------------------
    # HEART DISEASE PREDICTION
    # -----------------------------------------------------

    if check_heart_risk:

        try:

            resting_bp = float(heart_bp_input)

            cholesterol = float(cholesterol_input)

            max_heart_rate = float(max_heart_rate_input)

            if resting_bp < 1 or resting_bp > 250:
                st.error("Please enter a valid resting blood pressure between 1 and 250 mmHg.")
                st.stop()

            if cholesterol < 1 or cholesterol > 700:
                st.error("Please enter a valid cholesterol value between 1 and 700 mg/dL.")
                st.stop()
            
            if max_heart_rate < 40 or max_heart_rate > 250:
                st.error("Please enter a valid maximum heart rate between 40 and 250 bpm.")
                st.stop()


            # -------------------------------------------------
            # Convert UI values to dataset values
            # -------------------------------------------------

            sex = 1 if sex_input == "Male" else 0

            chest_pain_mapping = {
                "Typical angina": 1,
                "Atypical angina": 2,
                "Non-anginal pain": 3,
                "Asymptomatic": 4
            }

            chest_pain = chest_pain_mapping[chest_pain_input]

            exercise_angina = (
                1 if exercise_angina_input == "Yes" else 0
            )


            # -------------------------------------------------
            # Create patient dataframe
            # -------------------------------------------------

            patient = pd.DataFrame([{

                "Sex": sex,

                "ChestPainType": chest_pain,

                "RestingBloodPressure": resting_bp,

                "Cholesterol": cholesterol,

                "MaxHeartRate": max_heart_rate,

                "ExerciseInducedAngina": exercise_angina

            }])


            # -------------------------------------------------
            # Model prediction
            # -------------------------------------------------

            prediction = heart_model.predict(patient)[0]

            probability = heart_model.predict_proba(patient)[0][1]

            risk_percentage = probability * 100


            # -------------------------------------------------
            # Category
            # -------------------------------------------------

            if prediction == 1:

                category = "Higher Risk"
                category_class = "higher-risk"

            else:

                category = "Lower Risk"
                category_class = "lower-risk"


            # -------------------------------------------------
            # Result card
            # -------------------------------------------------

            result_html = f"""
<div class="result-card">

    <div class="result-column">

        <div class="result-label">
            Estimated Risk
        </div>

        <div class="result-value">
            {risk_percentage:.0f}%
        </div>

    </div>

    <div class="result-column result-divider">

        <div class="result-label">
            Category
        </div>

        <div class="risk-category {category_class}">
            ● {category}
        </div>

    </div>

</div>
"""

            st.html(result_html)


            # -------------------------------------------------
            # Gemini guidance
            # -------------------------------------------------

            try:

                guidance = generate_health_guidance(
                    disease,
                    risk_percentage,
                    category,
                    patient.to_dict(orient="records")[0]
                )

                if guidance:

                    guidance_html = f"""
<div class="guidance-card">
    {guidance.replace(chr(10), "<br>")}
</div>
"""

                    st.markdown(
                        guidance_html,
                        unsafe_allow_html=True
                    )

                else:

                    st.info(
                        "Additional health guidance is temporarily unavailable."
                    )

            except Exception:

                st.info(
                    "The risk assessment was completed, "
                    "but additional health guidance is temporarily unavailable."
                )


        except ValueError:

            st.error(
                "Please enter valid numeric values in all fields."
            )

        except Exception as e:

            st.error(
                f"Something went wrong while making the assessment: {e}"
            )