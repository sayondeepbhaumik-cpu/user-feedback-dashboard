import streamlit as st
import pandas as pd
from app import NLPEngine, ProfileScoringEngine, FeedbackLearner

# Set up page configurations
st.set_page_config(page_title="Intelligent Hybrid Matcher", page_icon="🎯", layout="wide")

st.title("🎯 Intelligent Hybrid Recommendation System")
st.subheader("Interactive User Profile Matching & AI Feedback Engine")

# Helper function to safely read fresh data metrics
@st.cache_data
def load_data():
    try:
        users = pd.read_csv("users.csv")
        feedback = pd.read_csv("feedback.csv")
        return users, feedback
    except FileNotFoundError:
        st.error("❌ Data files not found! Please run your backend engine script (app.py) first to generate them.")
        return None, None

df_users, df_feedback = load_data()

if df_users is not None:
    # Initialize the engine components built in app.py
    nlp_engine = NLPEngine(df_users)
    scoring_engine = ProfileScoringEngine(df_users, nlp_engine)
    feedback_learner = FeedbackLearner(df_feedback, scoring_engine)

    # Sidebar Navigation Option Panels
    st.sidebar.header("📁 Navigation Dashboard")
    option = st.sidebar.radio("Go to:", ["View User Records", "Calculate Match & Adapt System", "Top 5 Recommendations"])

    # ----------------------------------------------------
    # SECTION 1: VIEW USER RECORDS
    # ----------------------------------------------------
    if option == "View User Records":
        st.header("👥 User Registry Databank (users.csv)")
        st.write(f"Displaying registered profile details for *{len(df_users)} active members*.")
        st.dataframe(df_users, use_container_width=True)
        
        st.markdown("---")
        st.header("🔄 Historical Interaction Logs (feedback.csv)")
        st.write(f"Total tracking database size: *{len(df_feedback)} logged actions* (1 = Accept, 0 = Reject).")
        st.dataframe(df_feedback, use_container_width=True)

    # ----------------------------------------------------
    # SECTION 2: CALCULATE MATCH & ADAPT SYSTEM
    # ----------------------------------------------------
    elif option == "Calculate Match & Adapt System":
        st.header("🧠 Dynamic Multi-Layer Compatibility Test")
        st.write("Select two user profiles below to run cross-layer testing metrics.")

        col1, col2 = st.columns(2)
        with col1:
            user_a = st.selectbox("Select Target Profile (User A):", df_users['user_id'].unique())
            profile_a = df_users[df_users['user_id'] == user_a].iloc[0]
            st.info(f"*Name:* {profile_a['name']}\n\n*Role:* {profile_a['profession']}\n\n*MBTI:* {profile_a['mbti']} | *City:* {profile_a['location']}")
            
        with col2:
            user_b = st.selectbox("Select Candidate Profile (User B):", df_users['user_id'].unique(), index=1)
            profile_b = df_users[df_users['user_id'] == user_b].iloc[0]
            st.info(f"*Name:* {profile_b['name']}\n\n*Role:* {profile_b['profession']}\n\n*MBTI:* {profile_b['mbti']} | *City:* {profile_b['location']}")

        st.markdown("### ⚡ AI System Weight Optimization Mode")
        
        # Calculate baseline unoptimized uniform score configuration parameters
        baseline_score = scoring_engine.calculate_compatibility(user_a, user_b)
        
        # Pull adaptive training configuration models via feedback loops
        learned_weights = feedback_learner.optimize_weights_for_user(user_a)
        
        # Compute real-time personalized weight values
        adaptive_score = scoring_engine.calculate_compatibility(user_a, user_b, weights=learned_weights)

        # Visual layout display dashboards
        m1, m2 = st.columns(2)
        m1.metric(label="Baseline Profile Match (Unoptimized)", value=f"{baseline_score}%")
        m2.metric(label="Personalized Match (Adaptive ML Loop)", value=f"{adaptive_score}%", 
                  delta=f"{round(adaptive_score - baseline_score, 2)}% Shift")

        st.success(f"*System Update:* The feedback loop analyzed User A's logs and updated matching rules specifically for them to: "
                   f"Text Similarity (w1) = *{learned_weights['w1']}*, "
                   f"MBTI Matrix (w2) = *{learned_weights['w2']}*, "
                   f"Location Engine (w3) = *{learned_weights['w3']}*.")

    # ----------------------------------------------------
    # SECTION 3: TOP 5 RECOMMENDATIONS
    # ----------------------------------------------------
    elif option == "Top 5 Recommendations":
        st.header("🏆 Personalized Top 5 Match Recommendations")
        target_uid = st.selectbox("Select target member to generate recommendations for:", df_users['user_id'].unique())
        
        if st.button("Generate Ranked Recommendations"):
            with st.spinner("Processing NLP matrices and dynamic profile layers..."):
                user_weights = feedback_learner.optimize_weights_for_user(target_uid)
                
                all_candidates = []
                for _, row in df_users.iterrows():
                    other_id = row['user_id']
                    if other_id != target_uid:
                        score = scoring_engine.calculate_compatibility(target_uid, other_id, weights=user_weights)
                        all_candidates.append({
                            "Rank ID": other_id,
                            "Full Name": row['name'],
                            "Role/Profession": row['profession'],
                            "City Location": row['location'],
                            "MBTI Type": row['mbti'],
                            "Compatibility Score": f"{score}%"
                        })
                
                # Rank listings based on calculated values
                ranked_df = pd.DataFrame(all_candidates).sort_values(by="Compatibility Score", ascending=False).head(5)
                ranked_df.index = range(1, 6)
                
                st.balloons()
                st.write("Below are the top 5 highest-ranked profiles specifically calculated using this user's learned preference weights:")
                st.table(ranked_df)