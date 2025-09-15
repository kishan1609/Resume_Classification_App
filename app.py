import streamlit as st
import pickle

# -----------------------------
# Load Model and Vectorizer
# -----------------------------
@st.cache_resource
def load_model():
    with open("decision_tree_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfv = pickle.load(f)
    return model, tfv

model, tfv = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Resume Classification (Decision Tree)")

st.write("This app classifies resumes into categories using a Decision Tree model.")

# User input
resume_text = st.text_area("Paste Resume Text Here:")

if st.button("Predict"):
    if resume_text.strip():
        vectorized_text = tfv.transform([resume_text])
        prediction = model.predict(vectorized_text)[0]
        st.success(f"✅ Predicted Category: **{prediction}**")
    else:
        st.warning("⚠️ Please enter some resume text.")
