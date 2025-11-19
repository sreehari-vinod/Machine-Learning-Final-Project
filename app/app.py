"""
Personality Prediction Web Application
Author: Sreehari Vinod
Description: ML-powered web app to predict Introvert/Extrovert personality
             based on behavioral patterns using Random Forest Classifier
"""

import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Personality Prediction | ML App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    .css-1d391kg {
        padding-top: 2rem;
    }
    h1 {
        color: #2c3e50;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained machine learning model"""
    try:
        with open('model_personality.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found! Please ensure 'model_personality.pkl' exists in 'models/' directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


def main():
    """Main application function"""

    # Load model
    model = load_model()
    if model is None:
        st.stop()

    # Header
    st.title("🧠 Personality Prediction System")
    st.markdown("""
    <p style='font-size: 18px; color: #555;'>
    Discover your personality type using advanced Machine Learning algorithms. 
    This app analyzes your behavioral patterns to predict whether you're an <b>Introvert</b> or <b>Extrovert</b>.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # User information input
    col_name, col_age = st.columns([3, 1])
    with col_name:
        user_name = st.text_input(
            "👤 Full Name *",
            placeholder="e.g., John Doe",
            help="Enter your full name"
        )
    with col_age:
        user_age = st.number_input(
            "🎂 Age",
            min_value=13,
            max_value=100,
            value=25,
            help="Enter your age"
        )

    st.markdown("---")

    # Feature inputs in organized tabs
    tab1, tab2 = st.tabs(["📊 Social Behavior", "🎭 Personal Traits"])

    with tab1:
        st.markdown("### Social Interaction Patterns")
        col1, col2 = st.columns(2)

        with col1:
            time_alone = st.slider(
                "⏰ Time Spent Alone (hours/day)",
                min_value=0,
                max_value=10,
                value=5,
                help="How many hours per day do you prefer spending alone?"
            )

            social_events = st.slider(
                "🎉 Social Event Attendance (per month)",
                min_value=0,
                max_value=10,
                value=5,
                help="How many social events/gatherings do you attend monthly?"
            )

        with col2:
            going_outside = st.slider(
                "🚶 Going Outside Frequency (per week)",
                min_value=0,
                max_value=10,
                value=5,
                help="How often do you go outside for social activities per week?"
            )

            friends = st.slider(
                "👥 Friends Circle Size",
                min_value=0,
                max_value=20,
                value=10,
                help="Number of close friends you regularly interact with"
            )

    with tab2:
        st.markdown("### Personality Indicators")
        col3, col4 = st.columns(2)

        with col3:
            stage_fear = st.selectbox(
                "🎤 Do you have Stage Fear?",
                options=["No", "Yes"],
                help="Do you feel anxious speaking in front of groups?"
            )

            drained = st.selectbox(
                "😴 Feel Drained After Socializing?",
                options=["No", "Yes"],
                help="Do you feel exhausted after social interactions?"
            )

        with col4:
            post_freq = st.slider(
                "📱 Social Media Post Frequency (per week)",
                min_value=0,
                max_value=10,
                value=5,
                help="How often do you post on social media platforms?"
            )

    # Input summary expander
    with st.expander("📋 Review Your Input Summary", expanded=False):
        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.markdown("**Personal Info**")
            st.write(f"• Name: {user_name if user_name else 'Not provided'}")
            st.write(f"• Age: {user_age}")

        with summary_col2:
            st.markdown("**Social Behavior**")
            st.write(f"• Time Alone: {time_alone} hrs/day")
            st.write(f"• Social Events: {social_events}/month")
            st.write(f"• Going Outside: {going_outside}/week")
            st.write(f"• Friends: {friends}")

        with summary_col3:
            st.markdown("**Personal Traits**")
            st.write(f"• Stage Fear: {stage_fear}")
            st.write(f"• Drained After Socializing: {drained}")
            st.write(f"• Post Frequency: {post_freq}/week")

    st.markdown("---")

    # Convert categorical inputs
    stage_fear_encoded = 1 if stage_fear == "Yes" else 0
    drained_encoded = 1 if drained == "Yes" else 0

    # Create input dataframe
    input_data = pd.DataFrame({
        'Time_spent_Alone': [time_alone],
        'Stage_fear': [stage_fear_encoded],
        'Social_event_attendance': [social_events],
        'Going_outside': [going_outside],
        'Drained_after_socializing': [drained_encoded],
        'Friends_circle_size': [friends],
        'Post_frequency': [post_freq]
    })

    # Prediction button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 Predict My Personality", type="primary", use_container_width=True)

    if predict_button:
        if not user_name:
            st.warning("⚠️ Please enter your name before predicting!")
        else:
            with st.spinner("🔍 Analyzing your personality traits..."):
                # Make prediction
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)

                personality = "Introvert" if prediction[0] == 1 else "Extrovert"
                confidence = prediction_proba[0][prediction[0]] * 100

                # Display results
                st.markdown("---")
                st.markdown(f"## 🎯 Prediction Results for **{user_name}**")

                result_col1, result_col2 = st.columns([2, 1])

                with result_col1:
                    if personality == "Introvert":
                        st.success(f"### 🤫 {user_name}, you are likely an **{personality}**!")
                        st.info(f"**Confidence Level:** {confidence:.2f}%")
                        st.markdown("""
                        #### 📌 Introvert Characteristics:
                        - ✅ Prefer solitary activities and quiet environments
                        - ✅ Recharge energy through alone time
                        - ✅ Enjoy deep, meaningful one-on-one conversations
                        - ✅ Maintain smaller, close-knit friend circles
                        - ✅ Think before speaking and process internally
                        - ✅ Feel drained after prolonged social interactions
                        """)
                    else:
                        st.success(f"### 🎉 {user_name}, you are likely an **{personality}**!")
                        st.info(f"**Confidence Level:** {confidence:.2f}%")
                        st.markdown("""
                        #### 📌 Extrovert Characteristics:
                        - ✅ Energized by social interactions and group activities
                        - ✅ Enjoy meeting new people and networking
                        - ✅ Maintain wide social circles
                        - ✅ Outgoing, expressive, and spontaneous
                        - ✅ Think out loud and process externally
                        - ✅ Feel recharged after socializing
                        """)

                with result_col2:
                    # Confidence gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        title={'text': "Confidence Score", 'font': {'size': 20}},
                        number={'suffix': "%", 'font': {'size': 40}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1},
                            'bar': {'color': "#667eea" if personality == "Introvert" else "#f093fb"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 50], 'color': '#ffebee'},
                                {'range': [50, 75], 'color': '#fff3e0'},
                                {'range': [75, 100], 'color': '#e8f5e9'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={'color': "#2c3e50", 'family': "Arial"}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Probability distribution
                st.markdown("### 📊 Detailed Probability Analysis")
                prob_col1, prob_col2 = st.columns(2)

                with prob_col1:
                    prob_df = pd.DataFrame({
                        'Personality Type': ['Extrovert', 'Introvert'],
                        'Probability (%)': [
                            prediction_proba[0][0] * 100,
                            prediction_proba[0][1] * 100
                        ]
                    })
                    st.dataframe(
                        prob_df.style.format({'Probability (%)': '{:.2f}%'}),
                        use_container_width=True,
                        hide_index=True
                    )

                with prob_col2:
                    st.bar_chart(prob_df.set_index('Personality Type'))

                # Generate downloadable report
                st.markdown("---")
                st.markdown("### 📥 Download Your Personality Report")

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result_data = input_data.copy()
                result_data.insert(0, 'Name', user_name)
                result_data.insert(1, 'Age', user_age)
                result_data['Predicted_Personality'] = personality
                result_data['Confidence (%)'] = round(confidence, 2)
                result_data['Prediction_Date'] = timestamp

                csv = result_data.to_csv(index=False)

                st.download_button(
                    label="📄 Download Detailed Report (CSV)",
                    data=csv,
                    file_name=f"personality_report_{user_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # Sidebar information
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/brain.png", width=100)
        st.markdown("## 📌 About This App")
        st.markdown("""
        This application uses **Machine Learning** to predict personality types 
        based on behavioral and social interaction patterns.
        """)

        st.markdown("---")
        st.markdown("### 🎯 Model Performance")
        st.metric("Accuracy", "93%", "+2%")
        st.metric("Precision", "94%")
        st.metric("Recall", "92%")

        st.markdown("---")
        st.markdown("### 🤖 Technical Details")
        st.markdown("""
        - **Algorithm:** Random Forest Classifier
        - **Features:** 7 behavioral indicators
        - **Training Data:** 2,900+ samples
        - **Framework:** Scikit-learn
        """)

        # Feature importance visualization
        if hasattr(model, 'feature_importances_'):
            st.markdown("---")
            st.markdown("### 📊 Top Features")
            feature_imp = pd.DataFrame({
                'Feature': [
                    'Time Alone', 'Stage Fear', 'Social Events',
                    'Going Outside', 'Drained', 'Friends', 'Posts'
                ],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            st.bar_chart(feature_imp.set_index('Feature'))

        st.markdown("---")
        st.markdown("### 👨‍💻 Developer")
        st.markdown("""
        **Sreehari Vinod**  
        Data Science Student  
        [LinkedIn](https://linkedin.com) | [GitHub](https://github.com)
        """)

        st.markdown("---")
        st.caption("© 2025 Personality Prediction System | v1.0")


if __name__ == "__main__":
    main()
