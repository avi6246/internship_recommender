import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Set page config
st.set_page_config(
    page_title="Internship Recommender", 
    page_icon="🧑‍💻",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stSelectbox, .stMultiselect {
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .success-recommendation {
        padding: 20px;
        background-color: #e8f5e9;
        border-radius: 5px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🎓 Virtual Internship Recommender System")
st.markdown("""
    Fill your details below to get the best internship suggestion based on your profile!
    """)

# Sidebar for additional info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This recommendation system suggests internships based on:
    - Your education level
    - Your skills
    - Your interest domain
    """)
    st.markdown("---")
    st.markdown("### How it works")
    st.info("This system recommends internships based on your education, skills, and interests")
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")

# Main form
col1, col2 = st.columns(2)

with col1:
    education = st.selectbox(
        "Select your education level",
        ["B.Tech", "B.Sc", "BA", "BCA", "B.Com", "M.Tech", "M.Sc", "MBA"]
    )

with col2:
    interest = st.selectbox(
        "Select your interest domain", 
        [
            "Data Science", "Data Analytics", "HR", "Web Development", 
            "Finance", "Software Dev", "AI", "Marketing", "Cybersecurity"
        ]
    )

skills = st.multiselect(
    "Select your skills (select as many as applicable)",
    [
        "Python", "ML", "Excel", "Statistics", "Communication", 
        "Web", "Finance", "C++", "DSA", "Data", "Java", "SQL",
        "Tableau", "PowerBI", "JavaScript", "HTML/CSS", "React",
        "Angular", "Node.js", "AWS", "Azure", "Docker", "Git"
    ]
)

# Recommendation button
if st.button("Get Recommendation", key="recommend"):
    try:
        # Load the trained model and encoders
        model = joblib.load("model.pkl")
        mlb = joblib.load("skills_encoder.pkl")
        columns = joblib.load("model_columns.pkl")
        
        # Transform skills using MultiLabelBinarizer
        skill_data = mlb.transform([skills])
        skill_df = pd.DataFrame(skill_data, columns=mlb.classes_)
        
        # Create input features for education and interest
        input_dict = {
            f"Education_{education}": 1,
            f"Interest_{interest}": 1
        }
        input_df = pd.DataFrame([input_dict])
        
        # Combine all features
        full_df = pd.concat([input_df, skill_df], axis=1)
        
        # Add missing columns with 0 values
        for col in columns:
            if col not in full_df.columns:
                full_df[col] = 0
        
        # Ensure columns are in correct order
        full_df = full_df[columns]
        
        # Make prediction
        prediction = model.predict(full_df)[0]
        
        # Display recommendation with styling
        st.markdown(
            f"""
            <div class="success-recommendation">
                <h3>✅ Recommended Internship:</h3>
                <h2>{prediction}</h2>
                <p>Based on your profile of:</p>
                <ul>
                    <li>Education: {education}</li>
                    <li>Skills: {', '.join(skills)}</li>
                    <li>Interest: {interest}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please ensure all required model files (model.pkl, skills_encoder.pkl, model_columns.pkl) are in the correct directory.")