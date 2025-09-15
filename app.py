import streamlit as st
import pickle
import re

# Load the saved Decision Tree model and TF-IDF vectorizer
with open("model_DT.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfv.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing function (keep it consistent with your notebook)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Streamlit app setup
st.set_page_config(page_title="Resume Classification (Decision Tree)", page_icon="🌳", layout="centered")

st.title("🌳 Resume Classification Project")
st.write("This app classifies resumes into categories using a **Decision Tree model**.")

# Input option: text or file
option = st.radio("Choose input method:", ("Paste Resume Text", "Upload .docx File"))

resume_text = ""
if option == "Paste Resume Text":
    resume_text = st.text_area("Enter Resume Text Below:", height=200)
elif option == "Upload .docx File":
    uploaded_file = st.file_uploader("Upload a resume (.docx format only)", type=["docx"])
    if uploaded_file:
        resume_text = uploaded_file.read().decode("utf-8")

# Prediction
if st.button("🔍 Classify Resume"):
    if resume_text.strip() != "":
        clean_text = preprocess_text(resume_text)
        vectorized_text = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized_text)[0]

        st.success(f"✅ Predicted Category: **{prediction}**")
    else:
        st.warning("⚠️ Please enter or upload some resume text.")

st.markdown("---")
st.caption("🚀 Deployed with Streamlit | Model: Decision Tree Classifier")
