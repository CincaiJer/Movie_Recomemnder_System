import streamlit as st
import pandas as pd
import numpy as np
import joblib
from surprise import SVD

# Load the datasets
@st.cache_data
def load_data():
    credit = pd.read_csv("credits.xls")
    title = pd.read_csv("titles.xls")
    user = pd.read_csv("user_interactions.xls")
    
    # Aggregate credit details before merging
    credit_agg = credit.groupby("id").agg({
        "person_id": list,  
        "name": list,      
        "character": list,  
        "role": list       
    }).reset_index()
    
    # Merge titles with aggregated credits
    combine = title.merge(credit_agg, on="id", how="left")
    
    # Merge user interactions with the combined titles+credits dataset
    final_data = user.merge(combine, left_on="id", right_on="id", how="left")
    
    return final_data

df = load_data()

# Load the trained recommendation model
@st.cache_resource
def load_model():
    return joblib.load("movie_recommender.joblib")  # Load your saved model

svd_model = load_model()

# Function to recommend movies for a user
def recommend_movies(user_id, model, df):
    all_movies = df['id'].unique()
    user_rated_movies = df[df['user_id'] == user_id]['id'].values
    unrated_movies = [movie for movie in all_movies if movie not in user_rated_movies]
    
    pred_ratings = [model.predict(user_id, movie).est for movie in unrated_movies]
    top_movies = np.argsort(pred_ratings)[-5:][::-1]  # Get top 5 recommendations
    
    return [unrated_movies[i] for i in top_movies]

# Streamlit UI
st.title("🎬 Movie Recommender System")

# Input user ID
user_id = st.number_input("Enter Your User ID:", min_value=1, step=1)

# Button to get recommendations
if st.button("Get Recommendations"):
    recommendations = recommend_movies(user_id, svd_model, df)
    st.write("**Recommended Movies:**")
    for i, movie in enumerate(recommendations, start=1):
        st.write(f"{i}. {movie}")
