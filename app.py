import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title(" AI Review Sentiment Analyzer")

user_input = st.text_area("Enter your review:")

if st.button("Predict"):
    if user_input:
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)[0]

        if prediction == 1:
            st.success("Positive Review ")
        else:
            st.error("Negative Review ")
    else:
        st.warning("Please enter a review")