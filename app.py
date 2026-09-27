import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression

# ==========================================
# MODULE 0: DATASET CREATION ENGINE
# ==========================================

def generate_synthetic_datasets(num_users=60):
    """Generates realistic users.csv and feedback.csv datasets according to strict guidelines."""
    
    professions = ["Data Analyst", "Software Engineer", "Product Manager", "UX Designer", "Marketing Strategist"]
    locations = ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune"]
    mbti_types = ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP", "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"]
    
    prof_summaries = {
        "Data Analyst": "Data analyst with 3 years of experience in healthcare analytics. Skilled in SQL, Python, and automated business reporting frameworks.",
        "Software Engineer": "Full-stack developer specializing in scalable distributed systems. Highly proficient in Python, cloud architecture, and microservices.",
        "Product Manager": "Results-driven Product Manager with a track record of scaling consumer apps. Experienced in Agile methodologies, roadmap design, and user growth data metrics.",
        "UX Designer": "User experience designer focused on creating accessible digital platforms. Experienced in wireframing, high-fidelity prototyping, and cross-functional user testing.",
        "Marketing Strategist": "Growth marketer specialized in digital campaigns and brand positioning. Data-driven mindset with expertise in SEO, SEM, and customer acquisition funnels."
    }
    
    about_me_options = [
        "I enjoy solving complex real-world puzzles, mentoring junior peers, and driving productivity in high-performing collaborative team spaces.",
        "Passionate about continuous technical learning, reading science fiction, and collaborating closely with design teams on creative products.",
        "Thrive in fast-paced startup cultures. Emphasize open communication, systematic organization, and regular peer alignment.",
        "Love exploring outdoor hiking trails, volunteering in community workshops, and designing highly empathetic human-centric user experiences.",
        "A highly analytical strategist who values direct accountability, deep research focus, and structural clarity in execution frameworks."
    ]
    
    interests_pool = ["Data", "Fitness", "Teaching", "Coding", "Design", "Gaming", "Photography", "Travel", "Writing", "Cooking"]

    # 1. Create users.csv data
    users_data = []
    for i in range(1, num_users + 1):
        uid = f"U{str(i).zfill(3)}"
        name = f"User_{i}"
        age = random.randint(22, 45)
        loc = random.choice(locations)
        prof = random.choice(professions)
        exp = random.randint(1, 12)
        mbti = random.choice(mbti_types)
        
        if prof in ["Data Analyst", "Software Engineer"]:
            mbti = random.choice(["INTJ", "INTP", "ISTJ", "ENTP"])
        elif prof == "Product Manager":
            mbti = random.choice(["ENTJ", "ENFJ", "INTJ"])
            
        summary = prof_summaries[prof]
        about = random.choice(about_me_options)
        interests = ", ".join(random.sample(interests_pool, 3))
        
        users_data.append([uid, name, age, loc, prof, exp, summary, about, mbti, interests])
        
    df_users = pd.DataFrame(users_data, columns=[
        "user_id", "name", "age", "location", "profession", 
        "experience_years", "professional_summary", "about_me", "mbti", "interests"
    ])
    df_users.to_csv("users.csv", index=False)
    print("✔️ Successfully generated users.csv")

    # 2. Create feedback.csv data
    feedback_data = []
    for i in range(1, num_users + 1):
        uid = f"U{str(i).zfill(3)}"
        interacted_targets = random.sample([f"U{str(k).zfill(3)}" for k in range(1, num_users + 1) if k != i], 6)
        
        for target in interacted_targets:
            action = random.choice([0, 1])  
            feedback_data.append([uid, target, action])
            
    df_feedback = pd.DataFrame(feedback_data, columns=["user_id", "matched_user_id", "action"])
    df_feedback.to_csv("feedback.csv", index=False)
    print(f"✔️ Successfully generated feedback.csv ({len(df_feedback)} interactions)")

# ==========================================
# MODULE 1: NLP & SEMANTIC ANALYSIS
# ==========================================

class NLPEngine:
    def __init__(self, df_users):
        self.df_users = df_users.copy()
        self.df_users['combined_text'] = (
            self.df_users['professional_summary'].fillna('') + " " + 
            self.df_users['about_me'].fillna('')
        )
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df_users['combined_text'])
        self.id_to_index = {uid: idx for idx, uid in enumerate(self.df_users['user_id'])}

    def get_text_similarity(self, user_id_a, user_id_b):
        if user_id_a not in self.id_to_index or user_id_b not in self.id_to_index:
            return 0.0
        idx_a = self.id_to_index[user_id_a]
        idx_b = self.id_to_index[user_id_b]

        sim = cosine_similarity(self.tfidf_matrix[idx_a], self.tfidf_matrix[idx_b])
        return float(sim[0][0])

# ==========================================
# MODULE 2: PROFILE SCORING ENGINE
# ==========================================

class ProfileScoringEngine:
    def __init__(self, df_users, nlp_engine):
        self.df_users = df_users.set_index("user_id")
        self.nlp_engine = nlp_engine
        
    def get_mbti_match_score(self, mbti_a, mbti_b):
        if mbti_a == mbti_b:
            return 0.8
        
        ideal_pairs = {
            "INTJ": "ENFP", "ENFP": "INTJ",
            "INTP": "ENTJ", "ENTJ": "INTP",
            "INFJ": "ENFJ", "ENFJ": "INFJ",
            "INFP": "ENFJ", "ENFJ": "INFP"
        }
        if ideal_pairs.get(mbti_a) == mbti_b:
            return 1.0
            
        shared_dimensions = sum(1 for char_a, char_b in zip(mbti_a, mbti_b) if char_a == char_b)
        return 0.25 * shared_dimensions

    def calculate_compatibility(self, user_id_a, user_id_b, weights={'w1': 0.5, 'w2': 0.3, 'w3': 0.2}):
        if user_id_a not in self.df_users.index or user_id_b not in self.df_users.index:
            return 0.0
            
        user_a = self.df_users.loc[user_id_a]
        user_b = self.df_users.loc[user_id_b]
        
        text_sim = self.nlp_engine.get_text_similarity(user_id_a, user_id_b)
        mbti_match = self.get_mbti_match_score(user_a['mbti'], user_b['mbti'])
        location_match = 1.0 if user_a['location'] == user_b['location'] else 0.0
        
        total_score = (
            (weights['w1'] * text_sim) + 
            (weights['w2'] * mbti_match) + 
            (weights['w3'] * location_match)
        )
        
        return round(min(max(total_score * 100, 0.0), 100.0), 2)

# ==========================================
# MODULE 3: ADAPTIVE FEEDBACK LOOP
# ==========================================

class FeedbackLearner:
    def __init__(self, df_feedback, scoring_engine):
        self.df_feedback = df_feedback
        self.engine = scoring_engine

    def optimize_weights_for_user(self, target_user_id):
        user_history = self.df_feedback[self.df_feedback['user_id'] == target_user_id]
        
        if len(user_history) < 3 or user_history['action'].nunique() < 2:
            return {'w1': 0.5, 'w2': 0.3, 'w3': 0.2}
            
        features = []
        labels = []
        
        for _, row in user_history.iterrows():
            m_id = row['matched_user_id']
            action = row['action']
            
            u_a = self.engine.df_users.loc[target_user_id]
            u_b = self.engine.df_users.loc[m_id]
            
            t_sim = self.engine.nlp_engine.get_text_similarity(target_user_id, m_id)
            m_match = self.engine.get_mbti_match_score(u_a['mbti'], u_b['mbti'])
            l_match = 1.0 if u_a['location'] == u_b['location'] else 0.0
            
            features.append([t_sim, m_match, l_match])
            labels.append(action)
            
        model = LinearRegression(positive=True) 
        model.fit(features, labels)
        
        raw_weights = model.coef_
        total_weight = sum(raw_weights) if sum(raw_weights) > 0 else 1.0
        
        w1 = float(raw_weights[0] / total_weight) if total_weight > 0 else 0.33
        w2 = float(raw_weights[1] / total_weight) if total_weight > 0 else 0.33
        w3 = float(raw_weights[2] / total_weight) if total_weight > 0 else 0.34
        
        return {
            'w1': round(w1, 2),
            'w2': round(w2, 2),
            'w3': round(w3, 2)
        }

# ==========================================
# EXECUTION PIPELINE
# ==========================================

if __name__ == "__main__":
    print("--- STEP 1: INITIALIZING SYSTEM GENERATION PIPELINE ---")
    generate_synthetic_datasets(num_users=60)
    
    df_users = pd.read_csv("users.csv")
    df_feedback = pd.read_csv("feedback.csv")
    
    print("\n--- STEP 2: SPINNING UP COMPUTATION IMPLEMENTATION ENGINES ---")
    nlp_engine = NLPEngine(df_users)
    scoring_engine = ProfileScoringEngine(df_users, nlp_engine)
    feedback_learner = FeedbackLearner(df_feedback, scoring_engine)
    
    test_user = "U001"
    match_candidate = "U012"
    
    base_score = scoring_engine.calculate_compatibility(test_user, match_candidate)
    print(f"\nBaseline Compatibility Score between {test_user} and {match_candidate}: {base_score}%")
    
    print(f"\n--- STEP 3: EXECUTING RECOGNITION FEEDBACK OPTIMIZATION FOR {test_user} ---")
    optimized_weights = feedback_learner.optimize_weights_for_user(test_user)
    print(f"Dynamically Optimized Configuration Weights calculated for {test_user}: {optimized_weights}")